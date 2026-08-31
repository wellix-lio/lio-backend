# Lio v1

Lio هو وكيل ذكاء اصطناعي شخصي وتجاري متعدد اللغات، مصمم للعمل على الهاتف مع خادم آمن في الخلفية.

## ما الموجود في هذه الحزمة الآن؟
- تطبيق هاتف React Native / Expo.
- خادم FastAPI.
- نقطة محادثة `/chat`.
- ذاكرة محلية أولية SQLite.
- نظام موافقات أولي للعمليات الحساسة.
- بنية لوكلاء متخصصين: Research / Business / Communication / Monitoring.
- دعم تلقائي للعربية والإنجليزية والألمانية عبر تعليمات Lio.
- Web Search جاهز عند تفعيل OpenAI API.
- لا يوجد أي API Key داخل تطبيق الهاتف.

## البنية
mobile -> Lio Backend -> OpenAI Agents SDK -> Tools / Memory / Approvals

## تشغيل الخادم
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
# ضع OPENAI_API_KEY في .env على جهاز الخادم فقط عندما يصبح حساب API جاهزاً.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## تشغيل تطبيق الهاتف
ثبّت Node.js، ثم:
```bash
cd mobile
npm install
npx expo start
```

عدّل `EXPO_PUBLIC_LIO_API_URL` في ملف `.env` داخل مجلد mobile ليشير إلى عنوان الخادم.

مثال داخل الشبكة المحلية:
```env
EXPO_PUBLIC_LIO_API_URL=http://192.168.1.50:8000
```

## الأمان
لا تضع `OPENAI_API_KEY` في تطبيق الهاتف أو في Git.
المفتاح يجب أن يبقى على الخادم فقط.

## المرحلة التالية
1. تفعيل الصوت الفعلي STT/TTS.
2. تسجيل دخول آمن.
3. قاعدة بيانات سحابية.
4. Watch Center لمراقبة المواقع.
5. Gmail/Calendar/Files integrations.
6. إشعارات Push.
7. نشر Backend على Cloud.
8. بناء APK/Android production.


## تحديث v2
أضيفت واجهات Backend لتحويل الصوت إلى نص وتحويل جواب Lio إلى صوت. راجع `VOICE.md`.


## تحديث v3
أضيف أساس الذاكرة الدائمة، المشاريع، سجل التدقيق، الموافقات، وقاعدة Watch Center لمراقبة المواقع.


## تحديث v4
أضيف Task Center API وهيكل أقسام التطبيق للوصول لاحقاً إلى Android build.


## تحديث v5
أضيفت إعدادات نشر Docker وبناء Android APK/AAB، وتم تصحيح الموديل الافتراضي إلى `gpt-5.6-sol`.
