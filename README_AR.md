# Private Photo Studio

استوديو صور شخصي يعمل من داخل **GitHub Codespaces** عبر المتصفح، بدون Floot أو Replit.

## التشغيل

1. افتح المستودع.
2. اختر **Code → Codespaces → Create codespace on main**.
3. داخل الـCodespace شغّل:
   `bash scripts/setup_engines.sh`
4. بعدها شغّل:
   `python app.py`
5. افتح المنفذ **7860**.

## المحركات

- **FaceFusion** لتبديل الوجوه، اختيار الوجه المرجعي، المعالجة الجماعية، Face Enhancement وFrame Enhancement.
- **Real-ESRGAN** كمسار مستقل اختياري لتحسين الدقة.
- **Ollama Vision** اختياري لوصف الصور محليًا.
- **ComfyUI** مخطط له للتحرير التوليدي وinpainting/outpainting ونقل الوضعية.

FaceFusion الحالي يدعم نماذج تبديل متعددة، Pixel Boost حتى 1024×1024، Face Enhancer، وFrame Enhancer مثل Real-ESRGAN ×2/×4/×8، بالإضافة إلى مزودات CPU/CUDA/TensorRT/DirectML/ROCm/OpenVINO. citeturn974479search1turn974479search13turn974479search15turn974479search10

## تبديل الوجوه

الواجهة تدعم:
- صورة مرجعية واحدة.
- حتى 8 صور مستهدفة دفعة واحدة.
- اختيار نموذج تبديل الوجه.
- اختيار ترتيب الوجوه في الصورة المستهدفة.
- تحديد أي وجه في صورة المرجع يُستخدم.
- أوضاع جودة قياسية/عالية/فائقة.

## الجودة الفائقة

الوضع الفائق يركّب Face Swap + Face Enhancer + Frame Enhancer، مع حفظ الناتج بجودة عالية. هذا ليس Resize عاديًا؛ هو خط Super Resolution واستعادة للوجه عند توفر نماذج المحرك. citeturn974479search15

## ملاحظات مهمة

Codespaces العادي لا يضمن GPU. في حالة تشغيل نماذج ثقيلة بدون GPU سيكون الأداء أبطأ. FaceFusion يدعم CUDA وTensorRT على عتاد NVIDIA وخيارات أخرى حسب البيئة. citeturn974479search0turn974479search10
