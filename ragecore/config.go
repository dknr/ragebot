package ragecore

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds the bot's connection and storage settings. The recovery key is
// optional: it is only needed when the bot's own device has to be re-verified
// from server-side key backup (SSSS) and is not already verified.
type Config struct {
	Homeserver  string `json:"homeserver"`
	User        string `json:"user"`
	Password    string `json:"password"`
	RecoveryKey string `json:"recovery_key"`
	Database    string `json:"database"`
	PickleKey   string `json:"pickle_key"`
	DeviceName  string `json:"device_name"`
}

// LoadConfig reads the bot config from a JSON file, applying defaults for any
// field that is omitted.
func LoadConfig(path string) (*Config, error) {
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
		cfg.Database = "ragecore.db"
	}
	if cfg.PickleKey == "" {
		cfg.PickleKey = "ragecore"
	}
	if cfg.DeviceName == "" {
		cfg.DeviceName = "ragecore"
	}
	return &cfg, nil
}
