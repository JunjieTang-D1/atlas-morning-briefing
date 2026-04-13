FROM alpine:3.21 AS fetch
ARG VERSION=v1.1.2
# TARGETARCH is automatically set by Docker buildx (e.g. amd64, arm64)
ARG TARGETARCH=amd64
RUN apk add --no-cache curl ca-certificates \
    && curl -fsSL \
       "https://github.com/pufferffish/wireproxy/releases/download/${VERSION}/wireproxy_linux_${TARGETARCH}.tar.gz" \
       | tar -xz -C /usr/local/bin wireproxy \
    && chmod +x /usr/local/bin/wireproxy

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=fetch /usr/local/bin/wireproxy /usr/local/bin/wireproxy
ENTRYPOINT ["/usr/local/bin/wireproxy"]
