# Luna25-UET

## Usage

### Docker

Pull the image:

```bash
docker pull ghcr.io/dangdungcntt/luna25-uet
```

Run the container:

```bash
docker run -d -p 8000:8000 ghcr.io/dangdungcntt/luna25-uet
```

### Local

```bash
uv sync --locked
uv run api.py
```

### Environment variables

- `THRESHOLD`: Set the threshold for malignancy prediction (default: 0.7)

### API Endpoint

- `GET /docs`: OpenAPI documentation
- `POST /api/v1/predict/lesion`: Run inference on a CT scan
