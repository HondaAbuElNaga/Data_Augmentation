import json
import os
from os.path import join
import json_repair
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
from tqdm.auto import tqdm # عشان نشوف شريط تقدم
from dotenv import load_dotenv # <-- 1. إضافة مهمة

print("Starting Data Augmentation Step...")

# --- 1. إعداد OpenAI (مهم لتشغيل الخلية دي) ---
print("Setting up OpenAI client...")
try:
    # 2. تحميل المتغيرات من ملف .env
    load_dotenv() 
    
    # 3. جلب المفتاح من متغيرات البيئة
    # (تأكد أن الاسم في ملف .env هو OPENAI_API_KEY)
    api_key = os.getenv("OPENAI_API_KEY") 
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")

    openai_client = OpenAI(
        api_key=api_key,
    )
    openai_model_id = "gpt-4o-mini"
    print("OpenAI client ready.")
except Exception as e:
    print(f"Error setting up OpenAI client: {e}")
    print("Please make sure your .env file is in the same directory and contains the OPENAI_API_KEY.")


# --- 2. تعريف Pydantic Schema لـ "توليد الصياغات" ---
class TaskVariations(BaseModel):
    variations: List[str] = Field(
        ...,
        description="A list of 4 creative and diverse rephrasings of the original question.",
        min_items=4,
        max_items=4
    )

# --- 3. دالة (Function) لجلب الصياغات الجديدة ---
def get_question_variations(original_question: str, original_answer: str) -> List[str]:
    """
    يستخدم gpt-4o-mini لإنشاء صياغات بديلة للسؤال.
    """
    system_prompt = (
        "You are a data augmentation expert for LLM fine-tuning. "
        "Your job is to rephrase a given user question 4 different ways to create a diverse training dataset for a Q&A chatbot. "
        "The rephrased questions must ask for the exact same information as the original. "
        "Return *only* a JSON object matching the Pydantic schema, in the same language as the original question."
    )

    user_prompt = f"""
    The original question is:
    "{original_question}"

    The correct answer to this question is:
    "{original_answer}"

    Please generate 4 diverse rephrasings for the original question. Examples:
    - If original is "ما هي رؤية المعهد؟", variations could be "ما هي الرؤية الخاصة بالمعهد؟", "عن ماذا تنص رؤية المعهد؟", "أخبرني برؤية المعهد.", "ما هو الهدف الاستراتيجي أو رؤية المعهد؟".

    Return your answer as a JSON object matching this schema:
    {TaskVariations.model_json_schema()}
    """

    try:
        chat_completion = openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=openai_model_id,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        parsed_json = json_repair.loads(response_content)
        task_vars = TaskVariations(**parsed_json)
        return task_vars.variations
    except Exception as e:
        print(f"Warning: Error generating variations for question '{original_question}': {e}")
        return [] # رجع لستة فاضية لو حصل خطأ

# --- 4. تعريف Schema الإجابة (إللي عملناه قبل كده) ---
# (ده شكل الإجابة إللي الموديل هيتعلمها)
try:
    InstituteQA.model_json_schema
except NameError:
    class InstituteQA(BaseModel):
        answer: str = Field(..., description="The detailed and accurate answer.")
    print("Defined InstituteQA schema.")

# هنحتاج الـ Schema ده كـ "نص"
schema_json_str = json.dumps(InstituteQA.model_json_schema(), ensure_ascii=False)

# --- 5. الداتا الأساسية من الـ PDF ---
# (دي نفس الليستة من المرة إللي فاتت)
qa_data_from_pdf = [
    
    {"Q": "ايه هوه الزي في دبلوم اداره التمريض؟", "A": "الزي هو لاب كوت أو سكرب."},

    {"Q": "هل لو انا مسجل في معهد تاني اقدر اسجل معاكم؟", "A": "بالطبع لا، لا تستطيع التسجيل وأنت مسجل في معهد آخر."},

    {"Q": "هل الدورات التأهيلية ليها شهادة؟", "A": "نعم، لها شهادة. مدة الدورات التأهيلية تكون شهر، ولها اختبار، ويحصل المتدرب على شهادة اجتياز."},

    {"Q": "كم مدة الدورة التطويرية؟", "A": "مدتها 3 أيام ولها شهادة حضور."},

    {"Q": "هل الدورة التطويرية لها شهادة؟", "A": "نعم، لها شهادة حضور."},
    {"Q": "ما هي الدورات التأهيلية؟ ", "A": "الدورات هي: 1. استخدام الحاسب الالي في الأعمال المكتبية، 2. الإدارة المكتبية المتقدمة، 3. دورة الإدارة المكتبية (3 شهور)، 4. إدخال بيانات ومعالجة نصوص، 5. الارشفة الالكترونية المتقدمة (6 شهور)، 6. إدارة مكتبية (11 شهر)، 7. مقدمة في الارشفة الالكترونية (3 شهور)."},

    {"Q": "ما هي دورة استخدام الحاسب الالي في الأعمال المكتبية؟", "A": "دورة تأهيلية تهدف إلى تزويد المتدرب بالمهارات والمعلومات الأساسية اللازمة لاستخدام الحاسب في الأعمال المكتبية."},
    {"Q": "كم عدد الساعات المعتمدة لدورة استخدام الحاسب الالي؟", "A": "120 ساعة."},
    {"Q": "كم مدة دورة استخدام الحاسب الالي بالأشهر؟", "A": "3 أشهر."},
    {"Q": "كم مدة التدريب بالساعات في دورة استخدام الحاسب الالي؟", "A": "بواقع (120) ساعة تدريب."},
    {"Q": "على ماذا تعتمد دورة استخدام الحاسب الالي؟", "A": "تعتمد على مهارات حزمة برامج شركة مايكروسوفت المكتبية."},

    {"Q": "ما هي دورة الإدارة المكتبية المتقدمة؟", "A": "برنامج يتم فيه تأهيل المتدرب على القيام بمهام وإدارة الاعمال المكتبية المتقدمة."},
    {"Q": "كم عدد الساعات المعتمدة لدورة الإدارة المكتبية المتقدمة؟", "A": "360 ساعة."},
    {"Q": "كم مدة دورة الإدارة المكتبية المتقدمة؟", "A": "6 أشهر."},
    {"Q": "كم مدة التدريب بالساعات في دورة الإدارة المكتبية المتقدمة؟", "A": "بواقع (360) ساعة تدريب."},
    {"Q": "ما هي المهنة التي يتمكن الخريج من القيام بها بعد دورة الإدارة المكتبية المتقدمة؟", "A": "مهنة مدير مكتب."},

    {"Q": "ما هي دورة الإدارة المكتبية ؟", "A": "دورة تهدف لتنمية المعارف والاتجاهات والمهارات والخبرات الأساسية اللازمة التي تمكن المشارك من القيام بمهام الإدارة المكتبية."},
    {"Q": "كم عدد الساعات المعتمدة لدورة الإدارة المكتبية ؟", "A": "180 ساعة."},
    {"Q": "كم مدة دورة الإدارة المكتبية ؟", "A": "3 أشهر."},
    {"Q": "كم مدة التدريب بالساعات في دورة الإدارة المكتبية ؟", "A": "بواقع (180) ساعة تدريب."},
    {"Q": "ما هي المهنة التي يتمكن الخريج من القيام بها بعد دورة الإدارة المكتبية ؟", "A": "مهنة مدير مكتب."},

    {"Q": "ما هي دورة إدخال بيانات ومعالجة نصوص؟", "A": "دورة تأهيلية تهدف إلى تزويد المتدرب بالمهارات والمعلومات المتخصصة والاحترافية اللازمة لإدخال البيانات ومعالجة النصوص."},
    {"Q": "كم عدد الساعات المعتمدة لدورة إدخال بيانات ومعالجة نصوص؟", "A": "240 ساعة."},
    {"Q": "كم مدة دورة إدخال بيانات ومعالجة نصوص؟", "A": "6 أشهر."},
    {"Q": "كم مدة التدريب بالساعات في دورة إدخال بيانات ومعالجة نصوص؟", "A": "بواقع (240) ساعة تدريب."},
    {"Q": "ما الذي يتم التدرب عليه في دورة إدخال بيانات ومعالجة نصوص؟", "A": "التدرب على مهارات لوحة المفاتيح بطريقة اللمس، ثم معالجة النصوص باستخدام برنامج معالجة النصوص، ثم التدرب على المهارات الأساسية لبرنامج الجداول الإلكترونية من برامج شركة مايكروسوفت المكتبية."},

    {"Q": "ما هي دورة الارشفة الالكترونية المتقدمة؟", "A": "دورة تهدف إلى إكساب المتدرب المهارات والمعارف اللازمة في مقدمة في الأرشفة الإلكترونية."},
    {"Q": "كم عدد الساعات المعتمدة لدورة الارشفة الالكترونية المتقدمة؟", "A": "360 ساعة."},
    {"Q": "كم مدة دورة الارشفة الالكترونية المتقدمة؟", "A": "6 أشهر."},
    {"Q": "كم مدة التدريب بالساعات في دورة الارشفة الالكترونية المتقدمة؟", "A": "بواقع (360) ساعة تدريب."},
    {"Q": "ماذا يستطيع المتدرب ممارسته بنهاية دورة الارشفة الالكترونية المتقدمة؟", "A": "يستطيع المتدرب ممارسة الاعمال الأساسية في تقنيات رقمنة الوثائق الأرشيفية التي تساهم في زيادة الفعالية والإنتاجية للخدمات المقدمة في المؤسسات."},

    {"Q": "ما هي دورة إدارة مكتبية ؟", "A": "برنامج تدريبي يهدف لتنمية المعارف والاتجاهات والمهارات والخبرات الأساسية اللازمة التي تمكن المشارك من القيام بمهام الإدارة المكتبية ومهنة مدير المكتب."},
    {"Q": "كم عدد الساعات المعتمدة لدورة إدارة مكتبية ؟", "A": "660 ساعة."},
    {"Q": "كم مدة دورة إدارة مكتبية ؟", "A": "11 أشهر."},
    {"Q": "كم مدة التدريب بالساعات في دورة إدارة مكتبية ؟", "A": "بواقع (660) ساعة تدريب."},
    {"Q": "ما هي المهنة التي يمكن للخريج القيام بها بعد دورة إدارة مكتبية ؟", "A": "مهنة مدير المكتب."},
    {"Q": "ما هي نتائج التعلم المتوقعة من برنامج إدارة مكتبية ؟", "A": "تنمية المعارف والاتجاهات والمهارات والخبرات الأساسية اللازمة التي تمكن المشارك من القيام بمهام الإدارة المكتبية ومهنة مدير المكتب والتي تركز على أعمال الإدارة المكتبية."},

    {"Q": "ما هي دورة مقدمة في الارشفة الالكترونية (3 شهور)؟", "A": "دورة تهدف إلى إكساب المتدرب المهارات والمعارف اللازمة في مقدمة في الأرشفة الإلكترونية."},
    {"Q": "كم عدد الساعات المعتمدة لدورة مقدمة في الارشفة الالكترونية؟", "A": "180 ساعة."},
    {"Q": "كم مدة دورة مقدمة في الارشفة الالكترونية؟", "A": "3 أشهر."},
    {"Q": "كم مدة التدريب بالساعات في دورة مقدمة في الارشفة الالكترونية؟", "A": "بواقع (180) ساعة تدريب."},
    {"Q": "ماذا يستطيع المتدرب ممارسته بنهاية دورة مقدمة في الارشفة الالكترونية؟", "A": "يستطيع المتدرب ممارسة الاعمال الأساسية في تقنيات رقمنة الوثائق الأرشيفية التي تساهم في زيادة الفعالية والإنتاجية للخدمات المقدمة في المؤسسات."},
    {"Q": "ما هي الدورات التطويرية؟", "A": "الدورات هي: تطبيقات الذكاء الإصطناعي، تصميم المواقع والمتاجر الإلكترونية، الأرشفة الإلكترونية، مهارات الحاسب الالي للمستوى الاول، مهارات الحاسب الالي للمستوى الثاني، مهارات الحاسب الالي للمستوى الثالث، تطبيقات الحاسب الالي، تطبيقات الحاسب الالي الأساسية، تطبيقات الحاسب الالي المتقدمة، مقدمة في الحاسب الالي، التدريب علي مهارات الطباعة، إدارة الأولويات، دورة إستراتيجيات النجاح، المهارات الأساسية في الإدارة المكتبية، السلامة والصحة المهنية (OSHA)، إدارة المستودعات، ومهارات الأمن السيبراني."},
    {"Q": "ما هي التخصصات المتاح دراستها عن بعد؟", "A": "التخصص المتاح عن بعد هو دبلوم إدارة الموارد البشرية."},

    {"Q": "هل يمكنني دراسة دبلوم إدارة السلامة عن بعد؟", "A": "لا، هو دبلوم حضوري."},

    {"Q": "هل يمكنني دراسة دبلوم الأمن السيبراني عن بعد؟", "A": "لا، هو دبلوم حضوري."},

    {"Q": "هل يمكنني دراسة دبلوم القانون عن بعد؟", "A": "لا، هو دبلوم حضوري."},

    {"Q": "هل يمكنني دراسة دبلوم إدارة المستشفيات عن بعد؟", "A": "لا، هو دبلوم حضوري."},

    {"Q": "هل يمكنني دراسة دبلوم إدارة التمريض عن بعد؟", "A": "لا، هو دبلوم حضوري."},

    {"Q": "هل يمكنني دراسة دبلوم السلامة عن بعد؟", "A": "لا، هو دبلوم حضوري."},

]

# --- 6. عملية الـ Augmentation وحفظ الملف ---
print("Starting augmentation loop...")

# المسار إللي هنحفظ فيه (من تعريفك في الأول)
# <-- 4. تعديل المسار ليعمل بشكل سليم على ويندوز
data_dir = r"C:\Users\mohan\OneDrive\Desktop\Augmentation\data_dir"
sft_file_path = join(data_dir, "sft.jsonl")
os.makedirs(data_dir, exist_ok=True)

sft_records_to_write = [] # الليستة إللي هنجمع فيها الداتا
record_id = 0

# (استخدمنا tqdm عشان نشوف شريط التقدم)
for item in tqdm(qa_data_from_pdf, desc="Augmenting Q&A Pairs"):
    original_q = item["Q"]
    original_a = item["A"]

    # ده شكل الإجابة إللي الموديل هيطلعها (ثابتة)
    response_json_obj = {"answer": original_a}

    # 1. هات الصياغات الجديدة
    # (بنديله السؤال والجواب عشان يفهم السياق)
    new_questions = get_question_variations(original_q, original_a)

    # 2. ضيف السؤال الأصلي لليستة
    all_questions_for_this_answer = [original_q]

    # 3. ضيف الأسئلة الجديدة
    all_questions_for_this_answer.extend(new_questions)

    # 4. جهز كل الأسئلة دي عشان تتكتب في الملف
    # (بنعمل "تاسك" لكل سؤال جديد مع نفس الإجابة)
    for question_text in all_questions_for_this_answer:
        # ده الفورمات بتاع sft.jsonl زي النوت بوك الأصلية
        record = {
            "id": record_id,
            "story": "", # إحنا معندناش "قصة"، السؤال هو المدخل
            "task": question_text, # بنحط السؤال في خانة الـ "task"
            "output_scheme": schema_json_str, # الـ Schema بتاعنا
            "response": response_json_obj # الإجابة إللي هيتعلمها
        }
        sft_records_to_write.append(record)
        record_id += 1

print(f"\nAugmentation complete.")
print(f"Original questions: {len(qa_data_from_pdf)}")
print(f"Total augmented records (sft.jsonl size): {len(sft_records_to_write)}")

# --- 7. كتابة الملف ---
print(f"Writing {len(sft_records_to_write)} records to {sft_file_path}...")
with open(sft_file_path, "w", encoding="utf8") as f:
    for record in sft_records_to_write:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("sft.jsonl file created successfully!")