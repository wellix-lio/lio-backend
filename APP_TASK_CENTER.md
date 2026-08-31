# Lio v4 — App Shell & Task Center

هذه النسخة توسّع المشروع من مجرد شاشة محادثة إلى أساس تطبيق وكيل متكامل.

## أقسام التطبيق المستهدفة
- Home: المحادثة مع Lio.
- Tasks: المهمات التي ينفذها Lio وحالتها.
- Watch: المواقع/المنتجات التي تتم مراقبتها.
- Approvals: العمليات التي تنتظر موافقة المستخدم.
- Memory: المعلومات والمشاريع المحفوظة.
- Settings: اللغة والصوت والاتصالات.

## Task API
- POST /tasks
- GET /tasks/{user_id}

## حالة البناء
هذه حزمة source code وليست APK.
بناء APK production يحتاج Android build environment/signing أو خدمة build خارجية.
لا يتم وضع OpenAI API key داخل APK.
