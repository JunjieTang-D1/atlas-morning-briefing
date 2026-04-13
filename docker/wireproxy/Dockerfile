FROM golang:1.24-alpine AS build
RUN apk add --no-cache git \
    && git clone --depth=1 https://github.com/pufferffish/wireproxy.git /src \
    && cd /src \
    && CGO_ENABLED=0 go build -ldflags="-s -w" -o /usr/local/bin/wireproxy .

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=build /usr/local/bin/wireproxy /usr/local/bin/wireproxy
ENTRYPOINT ["/usr/local/bin/wireproxy"]
