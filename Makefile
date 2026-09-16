.PHONY: build run vet

# mautrix-go ships two olm implementations: the C libolm (needs the olm
# library installed) and the pure Go goolm. The goolm tag avoids the cgo
# dependency, so it is used by default here.
build:
	go build -tags goolm -o ragebot .

run:
	go run -tags goolm .

vet:
	go vet -tags goolm ./...
