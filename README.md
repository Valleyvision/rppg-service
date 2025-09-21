# from project root on host
sudo chown -R 1001:1001 recordings
docker compose down
docker compose up -d
curl "http://localhost:7676/metrics?duration=5&count=2"




curl -X POST "http://localhost:7676/questions" \
  -H "Content-Type: application/json" \
  -d '{"path":"recordings/metrics_f936779f35fd.json"}'



# rppg_vision
