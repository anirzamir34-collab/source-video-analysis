# Source Video Analysis v0.2 — real pose/tracking pipeline

Bu sürüm önceki `501 MODEL_PIPELINE_NOT_CONFIGURED` servisinin yerine gerçek video inference ekler.

## Gerçekte ne yapıyor?

- Video dosyasını geçici diske **stream** eder; tamamını RAM'e almak zorunda değildir.
- OpenCV ile videonun süresini ve hareket yoğunluğunu ölçer.
- Önce seyrek motion scan yapar, hareketli aralıkları daha sık örnekler.
- MediaPipe **Pose Landmarker Lite** ile kare başına en fazla 4 kişide 33 adet 3B pose landmark çıkarır.
- Kişileri zaman boyunca merkez, gövde boyutu ve pose geometrisiyle track eder.
- Ana karakteri zamansal süreklilik + görünür alan + merkezilik + pose görünürlüğü puanıyla seçer.
- Otur/kalk/çömel/uzan, kol kaldır-indir, el-yüz/gövde, kol uzat-geri çek, baş/gövde yön değişimi, yer değiştirme, el-kol hareketi ve NPC'ye yaklaşma/uzaklaşma/fiziksel örtüşme gibi **ölçülebilir** hareketleri zaman aralıklarına dönüştürür.
- Her action gerçek kaynak zaman kodu taşır ve `sourceVerified: true` yalnızca gözlenen pose/geometri kanıtından oluşturulan action'larda kullanılır.
- Sahne nesnesi, diyalog veya görünmeyen eylem uydurmaz.

## Önemli sınır

`mainMaleTrackId` alanı mevcut interaktif uygulamayla uyumluluk için korunmuştur. Bu motor **cinsiyet tahmini yapmaz**. Çok kişili sahnelerde ana karakter en kalıcı/büyük/merkezi pose track olarak seçilir. Yanlış kişi seçilen videolar için ileride manuel protagonist-lock eklenebilir.

## Endpoint'ler

- `GET /health`
- `GET /capabilities`
- `POST /analyze` — multipart `video`
- `POST /analyze-segment` — multipart `video`, opsiyonel `startTime`, `endTime`

Başarılı `/analyze` cevabı doğrudan `source-video-interactive-app` tarafından kullanılabilecek `actions[]`, `videoPrompt`, `semanticVideoMap`, `videoDuration`, `mainMaleTrackId` alanlarını döndürür.

## Render Free notu

512 MB RAM / 0.1 CPU üzerinde **Lite** pose modeli seçilmiştir. Uzun videolar yavaş olabilir. Sistem videonun tamamını her 0.25 saniyede taramak yerine iki aşamalı örnekleme kullanır: önce motion scan, sonra hareketli pencerelerde daha sık pose analizi. Render web servisleri uzun HTTP yanıtlarına izin verse de ön yüzdeki upstream timeout uzun videolar için ayrıca yükseltilmelidir.

## Deploy

Mevcut `source-video-analysis` GitHub reposundaki dosyaları bu paketle değiştirip push edin. Render bağlıysa otomatik deploy başlar.

Deploy sonrası:

1. `https://source-video-analysis.onrender.com/health`
2. `https://source-video-analysis.onrender.com/capabilities`

`capabilities` içinde şu değerler `true` olmalıdır:

```json
{
  "external_analysis_configured": true,
  "tracking": true,
  "whole_body_pose": true,
  "temporal_action_localization": true
}
```

Ardından önce 10–30 saniyelik basit bir videoyla `/analyze` test edin.
