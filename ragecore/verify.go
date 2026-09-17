package ragecore

import (
	"context"
	"fmt"

	"github.com/rs/zerolog"
	"maunium.net/go/mautrix/crypto"
	"maunium.net/go/mautrix/crypto/verificationhelper"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"
)

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
