package main

import (
	"context"
	"embed"
	"errors"
	"io/fs"
	"math/rand"
	"path"

	"maunium.net/go/mautrix"
	"maunium.net/go/mautrix/event"
	"maunium.net/go/mautrix/id"
)

// The neat GIFs live in assets/ and are embedded so the binary stays
// self-contained when they are present. The directory (not a GIF glob) is
// embedded so a checkout with zero GIFs still builds: the committed
// assets/README.md sentinel guarantees the embed has at least one file, and
// sendNeatGif skips when no neat-*.gif files are found.
//
//go:embed assets
var neatFS embed.FS

// sendNeatGif uploads a randomly chosen neat GIF and sends it to the room as
// an animated image message.
func sendNeatGif(ctx context.Context, client *mautrix.Client, roomID id.RoomID) error {
	matches, err := fs.Glob(neatFS, "assets/neat-*.gif")
	if err != nil {
		return err
	}
	if len(matches) == 0 {
		return errors.New("no neat GIFs embedded")
	}
	gif := matches[rand.Intn(len(matches))]
	data, err := neatFS.ReadFile(gif)
	if err != nil {
		return err
	}
	name := path.Base(gif)
	up, err := client.UploadBytesWithName(ctx, data, "image/gif", name)
	if err != nil {
		return err
	}
	content := &event.MessageEventContent{
		MsgType: event.MsgImage,
		Body:    name,
		URL:     up.ContentURI.CUString(),
		Info: &event.FileInfo{
			MimeType:   "image/gif",
			Size:       len(data),
			IsAnimated: true,
		},
	}
	_, err = client.SendMessageEvent(ctx, roomID, event.EventMessage, content)
	return err
}
