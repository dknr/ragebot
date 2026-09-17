package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"go.mau.fi/util/exzerolog"

	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/crypto"
	"maunium.net/go/mautrix/crypto/cryptohelper"
	"maunium.net/go/mautrix/crypto/verificationhelper"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"
)

// Config is read from a JSON file. The recovery key is optional: it is only
// needed when the bot's own device has to be re-verified from server-side
// key backup (SSSS) and is not already verified.
type Config struct {
	Homeserver  string `json:"homeserver"`
	User        string `json:"user"`
	Password    string `json:"password"`
	RecoveryKey string `json:"recovery_key"`
	Database    string `json:"database"`
	PickleKey   string `json:"pickle_key"`
}

// senti is the shared sentiment analyzer, initialized at startup.
var senti *analyzer

// neatRe matches "neat", "neat!", or "neat." as a whole word, case-insensitive.
var neatRe = regexp.MustCompile(`(?i)\bneat[!\.]*\b`)

func loadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	if cfg.Homeserver == "" || cfg.User == "" || cfg.Password == "" {
		return nil, fmt.Errorf("homeserver, user and password are required in the config")
	}
	if cfg.Database == "" {
		cfg.Database = "ragebot.db"
	}
	if cfg.PickleKey == "" {
		cfg.PickleKey = "ragebot"
	}
	return &cfg, nil
}

// verificationCallbacks implements the verificationhelper callback interfaces
// and auto-accepts incoming device verification requests so other sessions
// can verify this bot's device.
type verificationCallbacks struct {
	log zerolog.Logger
	vh  *verificationhelper.VerificationHelper
}

func (c *verificationCallbacks) VerificationRequested(ctx context.Context, txnID id.VerificationTransactionID, from id.UserID, fromDevice id.DeviceID) {
	c.log.Info().
		Stringer("transaction_id", txnID).
		Stringer("from", from).
		Stringer("from_device", fromDevice).
		Msg("Received verification request, accepting")
	if err := c.vh.AcceptVerification(ctx, txnID); err != nil {
		c.log.Warn().Err(err).Stringer("transaction_id", txnID).Msg("Failed to accept verification")
	}
}

func (c *verificationCallbacks) VerificationReady(ctx context.Context, txnID id.VerificationTransactionID, otherDeviceID id.DeviceID, supportsSAS, supportsScanQRCode bool, qrCode *verificationhelper.QRCode) {
	c.log.Info().
		Stringer("transaction_id", txnID).
		Stringer("other_device", otherDeviceID).
		Bool("supports_sas", supportsSAS).
		Bool("supports_scan_qr", supportsScanQRCode).
		Msg("Verification ready")
}

func (c *verificationCallbacks) VerificationCancelled(ctx context.Context, txnID id.VerificationTransactionID, code event.VerificationCancelCode, reason string) {
	c.log.Warn().
		Stringer("transaction_id", txnID).
		Str("code", string(code)).
		Str("reason", reason).
		Msg("Verification cancelled")
}

func (c *verificationCallbacks) VerificationDone(ctx context.Context, txnID id.VerificationTransactionID, method event.VerificationMethod) {
	c.log.Info().
		Stringer("transaction_id", txnID).
		Str("method", string(method)).
		Msg("Verification done")
}

func (c *verificationCallbacks) ShowSAS(ctx context.Context, txnID id.VerificationTransactionID, emojis []rune, emojiDescriptions []string, decimals []int) {
	if len(emojis) > 0 {
		c.log.Info().
			Stringer("transaction_id", txnID).
			Any("emojis", emojis).
			Any("descriptions", emojiDescriptions).
			Msg("SAS shown for verification")
	} else {
		c.log.Info().
			Stringer("transaction_id", txnID).
			Any("decimals", decimals).
			Msg("SAS shown for verification")
	}
}

// verifySelf makes sure the bot's own device is cross-signing verified. If it
// is not, it verifies it either with the recovery key from the config or by
// generating fresh cross-signing keys (which yields a new recovery key).
func verifySelf(ctx context.Context, mach *crypto.OlmMachine, password, recoveryKey string, log *zerolog.Logger) error {
	hasKeys, isVerified, err := mach.GetOwnVerificationStatus(ctx)
	if err != nil {
		return fmt.Errorf("failed to get own verification status: %w", err)
	}
	if isVerified {
		log.Info().Msg("Device is already verified")
		return nil
	}
	if recoveryKey != "" {
		log.Info().Msg("Device is not verified, verifying with recovery key")
		if err := mach.VerifyWithRecoveryKey(ctx, recoveryKey); err != nil {
			return fmt.Errorf("failed to verify device with recovery key: %w", err)
		}
		log.Info().Msg("Device verified with recovery key")
		return nil
	}
	if hasKeys {
		log.Warn().Msg("Cross-signing keys exist but device is not verified and no recovery key is configured; cannot verify device")
		return nil
	}
	log.Info().Msg("No cross-signing keys found, generating new ones")
	recoveryKey, keysCache, err := mach.GenerateAndUploadCrossSigningKeysWithPassword(ctx, password, "")
	if err != nil {
		return fmt.Errorf("failed to generate and upload cross-signing keys: %w", err)
	}
	if err := mach.ImportCrossSigningKeys(keysCache.Seeds()); err != nil {
		return fmt.Errorf("failed to store generated cross-signing keys: %w", err)
	}
	if err := mach.SignOwnDevice(ctx, mach.OwnIdentity()); err != nil {
		return fmt.Errorf("failed to sign own device: %w", err)
	}
	if err := mach.SignOwnMasterKey(ctx); err != nil {
		return fmt.Errorf("failed to sign own master key: %w", err)
	}
	log.Warn().Str("recovery_key", recoveryKey).Msg("Generated new recovery key; save it in the config")
	return nil
}

// signOutOtherSessions deletes every device/session of the user except the
// current one. Deleting a device requires user-interactive auth, so the
// password is sent as the UIA stage.
func signOutOtherSessions(ctx context.Context, client *mautrix.Client, password string, log *zerolog.Logger) error {
	devices, err := client.GetDevicesInfo(ctx)
	if err != nil {
		return fmt.Errorf("failed to get devices: %w", err)
	}
	var others []id.DeviceID
	for _, dev := range devices.Devices {
		if dev.DeviceID != client.DeviceID {
			others = append(others, dev.DeviceID)
		}
	}
	if len(others) == 0 {
		log.Info().Msg("No other sessions to sign out")
		return nil
	}
	log.Info().Int("count", len(others)).Msg("Signing out other sessions")
	for _, deviceID := range others {
		if err := deleteDeviceWithUIA(ctx, client, deviceID, password); err != nil {
			log.Warn().Err(err).Stringer("device_id", deviceID).Msg("Failed to sign out session")
		} else {
			log.Info().Stringer("device_id", deviceID).Msg("Signed out session")
		}
	}
	return nil
}

func deleteDeviceWithUIA(ctx context.Context, client *mautrix.Client, deviceID id.DeviceID, password string) error {
	err := client.DeleteDevice(ctx, deviceID, nil)
	if err == nil {
		return nil
	}
	var httpErr mautrix.HTTPError
	if !errors.As(err, &httpErr) || !httpErr.IsStatus(http.StatusUnauthorized) {
		return err
	}
	var uiAuthResp mautrix.RespUserInteractive
	if err := json.Unmarshal([]byte(httpErr.ResponseBody), &uiAuthResp); err != nil {
		return fmt.Errorf("failed to decode UIA response: %w", err)
	}
	auth := &mautrix.ReqUIAuthLogin{
		BaseAuthData: mautrix.BaseAuthData{
			Type:    mautrix.AuthTypePassword,
			Session: uiAuthResp.Session,
		},
		User:     client.UserID.String(),
		Password: password,
	}
	return client.DeleteDevice(ctx, deviceID, &mautrix.ReqDeleteDevice[any]{Auth: auth})
}

func main() {
	configPath := flag.String("config", "config.json", "path to the configuration file")
	flag.Parse()

	cfg, err := loadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	log := zerolog.New(zerolog.NewConsoleWriter(func(w *zerolog.ConsoleWriter) {
		w.TimeFormat = time.StampMilli
	})).With().Timestamp().Logger()
	exzerolog.SetupDefaults(&log)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	client, err := mautrix.NewClient(cfg.Homeserver, "", "")
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create client")
	}
	client.Log = log

	helper, err := cryptohelper.NewCryptoHelper(client, []byte(cfg.PickleKey), cfg.Database)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to create crypto helper")
	}
	helper.LoginAs = &mautrix.ReqLogin{
		Type:                     mautrix.AuthTypePassword,
		Identifier:               mautrix.UserIdentifier{Type: mautrix.IdentifierTypeUser, User: cfg.User},
		Password:                 cfg.Password,
		InitialDeviceDisplayName: "ragebot",
	}
	if err = helper.Init(ctx); err != nil {
		log.Fatal().Err(err).Msg("Failed to init crypto helper")
	}
	client.Crypto = helper
	mach := helper.Machine()

	log.Info().
		Stringer("user_id", client.UserID).
		Stringer("device_id", client.DeviceID).
		Msg("Logged in")

	if err := verifySelf(ctx, mach, cfg.Password, cfg.RecoveryKey, &log); err != nil {
		log.Fatal().Err(err).Msg("Failed to verify own device")
	}

	if err := signOutOtherSessions(ctx, client, cfg.Password, &log); err != nil {
		log.Warn().Err(err).Msg("Failed to sign out other sessions")
	}

	// Load the embedded sentiment model and ONNX Runtime. The model is used to
	// classify incoming message bodies; results are logged, not reacted to.
	senti, err = newAnalyzer(512)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load sentiment analyzer")
	}
	log.Info().Int("vocab", int(senti.tk.VocabSize())).Msg("Sentiment analyzer loaded")

	// Accept incoming device verification requests (SAS).
	cb := &verificationCallbacks{log: log}
	vh := verificationhelper.NewVerificationHelper(client, mach, nil, cb, false, false, true)
	cb.vh = vh
	if err = vh.Init(ctx); err != nil {
		log.Fatal().Err(err).Msg("Failed to init verification helper")
	}

	syncer := client.Syncer.(*mautrix.DefaultSyncer)

	// Log every received event: messages, invites, state, to-device, everything.
	syncer.OnEvent(func(ctx context.Context, evt *event.Event) {
		evtLog := log.Info().
			Str("type", evt.Type.String()).
			Stringer("sender", evt.Sender).
			Stringer("room_id", evt.RoomID).
			Stringer("event_id", evt.ID)
		if evt.StateKey != nil {
			evtLog = evtLog.Str("state_key", *evt.StateKey)
		}
		evtLog = evtLog.
			Int("source", int(evt.Mautrix.EventSource)).
			Bool("was_encrypted", evt.Mautrix.WasEncrypted)
		evtLog.Any("content", evt.Content.Raw).Msg("Received event")
	})

	// Auto-join whenever the bot is invited to a room.
	syncer.OnEventType(event.StateMember, func(ctx context.Context, evt *event.Event) {
		if evt.GetStateKey() != client.UserID.String() {
			return
		}
		if evt.Content.AsMember().Membership != event.MembershipInvite {
			return
		}
		log.Info().
			Stringer("room_id", evt.RoomID).
			Stringer("inviter", evt.Sender).
			Msg("Received room invite, joining")
		if _, err := client.JoinRoomByID(ctx, evt.RoomID); err != nil {
			log.Error().Err(err).Stringer("room_id", evt.RoomID).Msg("Failed to join room after invite")
		} else {
			log.Info().Stringer("room_id", evt.RoomID).Msg("Joined room after invite")
		}
	})

	// Run sentiment analysis on every incoming message, log the result, and
	// react with an emoji when the confidence clears a threshold. Skip our own
	// sends so we do not re-analyze what we said. Matrix allows multiple
	// reactions per event, so lizard and neat messages still get analyzed.
	syncer.OnEventType(event.EventMessage, func(ctx context.Context, evt *event.Event) {
		if evt.Sender == client.UserID {
			return
		}
		body := evt.Content.AsMessage().Body
		if body == "" {
			return
		}
		res, err := senti.classify(body)
		if err != nil {
			log.Error().Err(err).
				Stringer("room_id", evt.RoomID).
				Stringer("event_id", evt.ID).
				Msg("Sentiment analysis failed")
			return
		}
		log.Info().
			Str("label", res.Label).
			Float32("confidence", res.Confidence).
			Any("scores", res.Scores).
			Int("tokens", res.Tokens).
			Float64("elapsed_ms", res.ElapsedMS).
			Stringer("room_id", evt.RoomID).
			Stringer("event_id", evt.ID).
			Msg("Sentiment analysis")
		if emoji := emojiForSentiment(res.Label, res.Confidence); emoji != "" {
			log.Debug().
				Str("emoji", emoji).
				Stringer("room_id", evt.RoomID).
				Stringer("event_id", evt.ID).
				Msg("Sending sentiment reaction")
			if _, err := client.SendReaction(ctx, evt.RoomID, evt.ID, emoji); err != nil {
				log.Error().Err(err).Stringer("room_id", evt.RoomID).Stringer("event_id", evt.ID).Msg("Failed to send reaction")
			}
		}
	})

	// React with a lizard emoji to every incoming message whose body mentions
	// "lizard". Skip our own sends to avoid reacting to ourselves.
	syncer.OnEventType(event.EventMessage, func(ctx context.Context, evt *event.Event) {
		if evt.Sender == client.UserID {
			return
		}
		body := evt.Content.AsMessage().Body
		if !strings.Contains(strings.ToLower(body), "lizard") {
			return
		}
		log.Debug().
			Stringer("room_id", evt.RoomID).
			Stringer("event_id", evt.ID).
			Msg("Sending lizard reaction")
		if _, err := client.SendReaction(ctx, evt.RoomID, evt.ID, "🦎"); err != nil {
			log.Error().Err(err).Stringer("room_id", evt.RoomID).Stringer("event_id", evt.ID).Msg("Failed to send reaction")
		}
	})

	// Send a random neat GIF whenever an incoming message says "neat" (also
	// "neat!" or "neat."). Skip our own sends to avoid GIFing ourselves.
	syncer.OnEventType(event.EventMessage, func(ctx context.Context, evt *event.Event) {
		if evt.Sender == client.UserID {
			return
		}
		body := evt.Content.AsMessage().Body
		if !neatRe.MatchString(body) {
			return
		}
		log.Debug().
			Stringer("room_id", evt.RoomID).
			Stringer("event_id", evt.ID).
			Msg("Sending neat GIF")
		if err := sendNeatGif(ctx, client, evt.RoomID); err != nil {
			log.Error().Err(err).Stringer("room_id", evt.RoomID).Stringer("event_id", evt.ID).Msg("Failed to send neat GIF")
		}
	})

	log.Info().Msg("Now running")

	syncCtx, cancelSync := context.WithCancel(ctx)
	defer cancelSync()
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := client.SyncWithContext(syncCtx)
		if err != nil && !errors.Is(err, context.Canceled) {
			log.Error().Err(err).Msg("Sync failed")
			cancelSync()
		}
	}()

	<-ctx.Done()
	cancelSync()
	wg.Wait()
	log.Info().Msg("Shutting down")
}
