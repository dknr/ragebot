package ragecore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/rs/zerolog"
	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/id"
)

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
