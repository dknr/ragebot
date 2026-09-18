package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"regexp"
	"strings"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"maunium.net/go/mautrix/event"

	"ragecore"
)

// senti is the shared sentiment analyzer, initialized at startup.
var senti *analyzer

// neatRe matches "neat", "neat!", or "neat." as a whole word, case-insensitive.
var neatRe = regexp.MustCompile(`(?i)\bneat[!\.]*\b`)

func main() {
	configPath := flag.String("config", "config.json", "path to the configuration file")
	flag.Parse()

	cfg, err := ragecore.LoadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load config: %v\n", err)
		os.Exit(1)
	}

	log := zerolog.New(zerolog.NewConsoleWriter(func(w *zerolog.ConsoleWriter) {
		w.TimeFormat = time.StampMilli
	})).With().Timestamp().Logger()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	bot, err := ragecore.New(ctx, cfg, log)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to start bot")
	}

	// Load the embedded sentiment model and ONNX Runtime. The model is used to
	// classify incoming message bodies; results are logged, not reacted to.
	senti, err = newAnalyzer(512)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load sentiment analyzer")
	}
	log.Info().Int("vocab", int(senti.tk.VocabSize())).Msg("Sentiment analyzer loaded")

	// Log every received event: messages, invites, state, to-device, everything.
	bot.OnEvent(func(ctx context.Context, evt *event.Event) {
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

	// Run sentiment + irony analysis on every incoming message, log both, and
	// react with an emoji when a confidence clears a threshold. Irony takes
	// priority over sentiment. OnMessage skips our own sends so we do not
	// re-analyze what we said. Matrix allows multiple reactions per event, so
	// lizard and neat messages still get analyzed.
	bot.OnMessage(func(ctx context.Context, evt *event.Event) {
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
		ironyScore := float32(0)
		irony, ierr := senti.classifyIrony(body)
		if ierr != nil {
			log.Error().Err(ierr).
				Stringer("room_id", evt.RoomID).
				Stringer("event_id", evt.ID).
				Msg("Irony analysis failed, degrading to sentiment-only")
		} else {
			ironyScore = irony.Scores[1]
		}
		log.Info().
			Str("label", res.Label).
			Float32("confidence", res.Confidence).
			Any("scores", res.Scores).
			Int("tokens", res.Tokens).
			Float64("elapsed_ms", res.ElapsedMS).
			Str("irony_label", irony.Label).
			Float32("irony_score", ironyScore).
			Stringer("room_id", evt.RoomID).
			Stringer("event_id", evt.ID).
			Msg("Sentiment analysis")
		if emoji := emojiForSentiment(res.Label, res.Confidence, ironyScore); emoji != "" {
			log.Debug().
				Str("emoji", emoji).
				Stringer("room_id", evt.RoomID).
				Stringer("event_id", evt.ID).
				Msg("Sending sentiment reaction")
			if _, err := bot.Client().SendReaction(ctx, evt.RoomID, evt.ID, emoji); err != nil {
				log.Error().Err(err).Stringer("room_id", evt.RoomID).Stringer("event_id", evt.ID).Msg("Failed to send reaction")
			}
		}
	})

	// React with a lizard emoji to every incoming message whose body mentions
	// "lizard". OnMessage skips our own sends to avoid reacting to ourselves.
	bot.OnMessage(func(ctx context.Context, evt *event.Event) {
		body := evt.Content.AsMessage().Body
		if !strings.Contains(strings.ToLower(body), "lizard") {
			return
		}
		log.Debug().
			Stringer("room_id", evt.RoomID).
			Stringer("event_id", evt.ID).
			Msg("Sending lizard reaction")
		if _, err := bot.Client().SendReaction(ctx, evt.RoomID, evt.ID, "🦎"); err != nil {
			log.Error().Err(err).Stringer("room_id", evt.RoomID).Stringer("event_id", evt.ID).Msg("Failed to send reaction")
		}
	})

	// Send a random neat GIF whenever an incoming message says "neat" (also
	// "neat!" or "neat."). OnMessage skips our own sends to avoid GIFing
	// ourselves.
	bot.OnMessage(func(ctx context.Context, evt *event.Event) {
		body := evt.Content.AsMessage().Body
		if !neatRe.MatchString(body) {
			return
		}
		log.Debug().
			Stringer("room_id", evt.RoomID).
			Stringer("event_id", evt.ID).
			Msg("Sending neat GIF")
		if err := sendNeatGif(ctx, bot.Client(), evt.RoomID); err != nil {
			log.Error().Err(err).Stringer("room_id", evt.RoomID).Stringer("event_id", evt.ID).Msg("Failed to send neat GIF")
		}
	})

	// Run emotion analysis on every incoming message, log the 11-class
	// probabilities, and react with the highest-scoring emotion emoji when it
	// clears the threshold. OnMessage skips our own sends to avoid reacting to
	// ourselves.
	bot.OnMessage(func(ctx context.Context, evt *event.Event) {
		body := evt.Content.AsMessage().Body
		if body == "" {
			return
		}
		res, err := senti.classifyEmotions(body)
		if err != nil {
			log.Error().Err(err).
				Stringer("room_id", evt.RoomID).
				Stringer("event_id", evt.ID).
				Msg("Emotion analysis failed")
			return
		}
		emojis := emojiForEmotion(res.Label, res.Confidence)
		log.Info().
			Str("label", res.Label).
			Float32("confidence", res.Confidence).
			Any("scores", res.Scores).
			Int("tokens", res.Tokens).
			Float64("elapsed_ms", res.ElapsedMS).
			Stringer("room_id", evt.RoomID).
			Stringer("event_id", evt.ID).
			Msg("Emotion analysis")
		if emojis != "" {
			log.Debug().
				Str("emoji", emojis).
				Stringer("room_id", evt.RoomID).
				Stringer("event_id", evt.ID).
				Msg("Sending emotion reaction")
			if _, err := bot.Client().SendReaction(ctx, evt.RoomID, evt.ID, emojis); err != nil {
				log.Error().Err(err).Stringer("room_id", evt.RoomID).Stringer("event_id", evt.ID).Msg("Failed to send emotion reaction")
			}
		}
	})

	if err := bot.Run(ctx); err != nil {
		log.Fatal().Err(err).Msg("Bot failed")
	}
}
