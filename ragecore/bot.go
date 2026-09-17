package ragecore

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/rs/zerolog"
	"go.mau.fi/util/exzerolog"

	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/crypto/cryptohelper"
	"maunium.net/go/mautrix/crypto/verificationhelper"
	"maunium.net/go/mautrix/event"
)

// Bot is a ready-to-run Matrix bot with crypto and verification set up.
// Create one with New, register handlers with OnEvent and OnMessage, then
// call Run to start the sync loop.
type Bot struct {
	client          *mautrix.Client
	helper          *cryptohelper.CryptoHelper
	log             zerolog.Logger
	cfg             *Config
	eventHandlers   []func(ctx context.Context, evt *event.Event)
	messageHandlers []func(ctx context.Context, evt *event.Event)
}

// New creates a Bot, logs in, and performs crypto setup: self-verification,
// signing out other sessions, and accepting incoming device verification
// requests. Handlers must be registered on the returned Bot before calling
// Run.
func New(ctx context.Context, cfg *Config, log zerolog.Logger) (*Bot, error) {
	exzerolog.SetupDefaults(&log)

	client, err := mautrix.NewClient(cfg.Homeserver, "", "")
	if err != nil {
		return nil, fmt.Errorf("failed to create client: %w", err)
	}
	client.Log = log

	helper, err := cryptohelper.NewCryptoHelper(client, []byte(cfg.PickleKey), cfg.Database)
	if err != nil {
		return nil, fmt.Errorf("failed to create crypto helper: %w", err)
	}
	helper.LoginAs = &mautrix.ReqLogin{
		Type:                     mautrix.AuthTypePassword,
		Identifier:               mautrix.UserIdentifier{Type: mautrix.IdentifierTypeUser, User: cfg.User},
		Password:                 cfg.Password,
		InitialDeviceDisplayName: cfg.DeviceName,
	}
	if err = helper.Init(ctx); err != nil {
		return nil, fmt.Errorf("failed to init crypto helper: %w", err)
	}
	client.Crypto = helper
	mach := helper.Machine()

	log.Info().
		Stringer("user_id", client.UserID).
		Stringer("device_id", client.DeviceID).
		Msg("Logged in")

	if err := verifySelf(ctx, mach, cfg.Password, cfg.RecoveryKey, &log); err != nil {
		return nil, fmt.Errorf("failed to verify own device: %w", err)
	}

	if err := signOutOtherSessions(ctx, client, cfg.Password, &log); err != nil {
		log.Warn().Err(err).Msg("Failed to sign out other sessions")
	}

	cb := &verificationCallbacks{log: log}
	vh := verificationhelper.NewVerificationHelper(client, mach, nil, cb, false, false, true)
	cb.vh = vh
	if err = vh.Init(ctx); err != nil {
		return nil, fmt.Errorf("failed to init verification helper: %w", err)
	}

	return &Bot{
		client: client,
		helper: helper,
		log:    log,
		cfg:    cfg,
	}, nil
}

// Run starts the sync loop and blocks until ctx is done, then shuts down
// gracefully. Register handlers with OnEvent and OnMessage before calling Run.
func (b *Bot) Run(ctx context.Context) error {
	syncer := b.client.Syncer.(*mautrix.DefaultSyncer)

	// Auto-join whenever the bot is invited to a room.
	syncer.OnEventType(event.StateMember, func(ctx context.Context, evt *event.Event) {
		if evt.GetStateKey() != b.client.UserID.String() {
			return
		}
		if evt.Content.AsMember().Membership != event.MembershipInvite {
			return
		}
		b.log.Info().
			Stringer("room_id", evt.RoomID).
			Stringer("inviter", evt.Sender).
			Msg("Received room invite, joining")
		if _, err := b.client.JoinRoomByID(ctx, evt.RoomID); err != nil {
			b.log.Error().Err(err).Stringer("room_id", evt.RoomID).Msg("Failed to join room after invite")
		} else {
			b.log.Info().Stringer("room_id", evt.RoomID).Msg("Joined room after invite")
		}
	})

	// Wire registered handlers.
	for _, fn := range b.eventHandlers {
		syncer.OnEvent(fn)
	}
	for _, fn := range b.messageHandlers {
		syncer.OnEventType(event.EventMessage, func(ctx context.Context, evt *event.Event) {
			if evt.Sender == b.client.UserID {
				return
			}
			fn(ctx, evt)
		})
	}

	b.log.Info().Msg("Now running")

	syncCtx, cancelSync := context.WithCancel(ctx)
	defer cancelSync()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := b.client.SyncWithContext(syncCtx)
		if err != nil && !errors.Is(err, context.Canceled) {
			b.log.Error().Err(err).Msg("Sync failed")
			cancelSync()
		}
	}()

	<-ctx.Done()
	cancelSync()
	wg.Wait()
	b.log.Info().Msg("Shutting down")
	return nil
}

// Client returns the underlying mautrix client for advanced operations.
func (b *Bot) Client() *mautrix.Client {
	return b.client
}

// OnEvent registers a handler for all incoming events: messages, invites,
// state, to-device, everything. Handlers must be registered before Run.
func (b *Bot) OnEvent(fn func(ctx context.Context, evt *event.Event)) {
	b.eventHandlers = append(b.eventHandlers, fn)
}

// OnMessage registers a handler for room messages from other users. The bot's
// own outgoing messages are not delivered. Handlers must be registered before
// Run.
func (b *Bot) OnMessage(fn func(ctx context.Context, evt *event.Event)) {
	b.messageHandlers = append(b.messageHandlers, fn)
}
