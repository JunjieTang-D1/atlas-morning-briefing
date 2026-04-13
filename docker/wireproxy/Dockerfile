FROM golang:1.24-alpine AS build
RUN apk add --no-cache git \
    && go install github.com/pufferffish/wireproxy@latest

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=build /go/bin/wireproxy /usr/local/bin/wireproxy
ENTRYPOINT ["/usr/local/bin/wireproxy"]
