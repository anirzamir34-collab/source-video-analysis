# Source Video Interactive — External Analysis Service

Bu paket, Google AI Studio uygulamasındaki `EXTERNAL_ANALYSIS_URL` alanına bağlanacak harici FastAPI servisinin **ilk güvenli sürümüdür**.

## Bu sürüm ne yapar?

- `GET /health` gerçek bağlantı testi yapar.
- `GET /capabilities` hangi analiz katmanlarının gerçekten kurulu olduğunu bildirir.
- `POST /analyze` ve `POST /analyze-segment` şimdilik **501 Not Implemented** döndürür.
- ByteTrack, MMPose/RTMPose veya MMAction2 kurulmuş gibi davranmaz.
- Sahte action/timestamp üretmez.

Amaç önce Google AI Studio ↔ harici servis bağlantısını kesin olarak doğrulamaktır.

## Yerelde çalıştırma

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl http://localhost:8000/health
```

Beklenen yanıt:

```json
{
  "status": "ok",
  "service": "video-analysis",
  "version": "0.1.0",
  "environment": "production"
}
```

## Docker ile çalıştırma

```bash
docker build -t source-video-analysis .
docker run --rm -p 8080:8080 source-video-analysis
```

## Google Cloud Run'a dağıtma

Google Cloud SDK kurulmuş ve oturum açılmışsa proje klasöründe:

```bash
gcloud run deploy source-video-analysis \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated
```

Dağıtım sonunda Cloud Run size şu tip bir HTTPS adresi verir:

```text
https://source-video-analysis-xxxxx.europe-west1.run.app
```

Önce `/health` adresini kontrol edin. `status: ok` döndükten sonra bu URL'yi Google AI Studio uygulamasındaki harici servis endpoint alanına yazın.

## Google AI Studio tarafında beklenen davranış

Bağlantı testi:

```text
GET {EXTERNAL_ANALYSIS_URL}/health
```

`200 OK` + `status=ok` alırsa servis durumu `CONNECTED` olarak gösterilebilir.

Ancak `/capabilities` şu aşamada analiz modellerinin kapalı olduğunu söyleyecektir. Bu nedenle uygulama gerçek ML analizi hazırmış gibi davranmamalıdır.

## Sonraki aşama

Bağlantı doğrulandıktan sonra ayrı ayrı eklenecek katmanlar:

1. Video upload ve güvenli geçici dosya yönetimi
2. Person detection / persistent tracking
3. MAIN_MALE_TRACK_ID kilitleme
4. Whole-body pose
5. RGB temporal action localization
6. start/end frame doğrulaması
7. Verified Master Action Timeline JSON
8. `/analyze` ve `/analyze-segment` gerçek inference

Bu katmanlar eklenene kadar servis bilinçli olarak sahte sonuç döndürmez.
