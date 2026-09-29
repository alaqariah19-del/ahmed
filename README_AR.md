# Private Photo Studio

استوديو صور شخصي يعمل من داخل **GitHub Codespaces** عبر المتصفح، بدون Floot أو Replit.

## التشغيل

1. افتح المستودع في GitHub.
2. اختر **Code → Codespaces → Create codespace on main**.
3. بعد إنشاء الـCodespace شغّل: `python app.py`
4. افتح المنفذ **7860** من تبويب Ports.

## الوظائف

- تبديل الوجه: وجه مرجعي واحد + عدة صور مستهدفة.
- اختيار ترتيب مطابقة الوجوه.
- تحسين الجودة عند توصيل Real-ESRGAN.
- واجهة موحدة لإضافة نقل الوضعية والتحرير والوصف والتركيب لاحقًا.
- الصور تبقى داخل بيئة التشغيل ولا تُرفع تلقائيًا لخدمة خارجية.

## المحركات

ضع مسار FaceFusion في المتغير `FACEFUSION_DIR` داخل Codespace عند توفره. ويمكن تحديد Real-ESRGAN عبر `REALESRGAN_EXE`.

> ملاحظة: Codespaces العادي لا يضمن GPU. تشغيل FaceFusion قد يكون بطيئًا أو يحتاج Runner مزودًا بـGPU إذا كان النموذج ثقيلًا.
