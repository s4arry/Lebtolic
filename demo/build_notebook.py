"""Regenerate LEPTOLIC_Demo.ipynb (run from repo root: python -m demo.build_notebook)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# LEPTOLIC — Demo ทดลองกรอกค่า

ฟอร์มสำหรับโมเดลอ้างอิง **befec00** (Logistic Regression C=0.1 · อายุ เพศ SBP DBP น้ำหนัก สูบบุหรี่)

**วิธีใช้:** `Runtime → Run all` แล้วเลื่อนลงไปที่ฟอร์ม กรอกค่า กด **ทำนาย**

ผลคือ *คะแนนทดลอง* ของโอกาสที่จะ **รายงานว่าได้รับการวินิจฉัยความดันสูงครั้งแรกภายใน 8 ปี**
จากข้อมูล NHEFS สหรัฐฯ ปี 1982–1992 — ไม่ใช่การวินิจฉัย และยังไม่ได้ทดสอบกับคนไทย"""),
    code("""# ถ้าเปิดใน Google Colab จะ clone repo มาให้อัตโนมัติ (รันบนเครื่องตัวเองข้ามได้)
from pathlib import Path
import os, subprocess
BRANCH = 'dev-by-gunner'
if not Path('model/model.json').exists():
    if not Path('Lebtolic').exists():
        subprocess.run(['git', 'clone', '-q', '-b', BRANCH,
                        'https://github.com/s4arry/Lebtolic.git'], check=True)
    os.chdir('Lebtolic')
print('working directory:', Path.cwd())"""),
    code("""from demo.form import Answers, InputRejected, PRESETS, SMOKING_TH, FEATURE_TH, load_model, predict

model, state = load_model()
THRESHOLD = state['threshold_policies']['fpr145']['threshold']
print(f'โหลดโมเดลแล้ว · {len(state["features"])} encoded columns · threshold {THRESHOLD:.2%}')"""),
    md("""## ฟอร์ม

- **ปุ่มตัวอย่าง** ต่ำ / กลาง / สูง เติมค่าสมมติให้ (ไม่ใช่คนจริง)
- **Guardrails:** ระบบจะปฏิเสธถ้าเคยวินิจฉัยแล้ว, ความดัน ≥ 130/80, หรือค่าอยู่นอกช่วงข้อมูลที่โมเดลเรียน"""),
    code("""import ipywidgets as W
from IPython.display import display, HTML

style = {'description_width': '150px'}
age = W.FloatText(value=45, description='อายุ (ปี)', style=style)
sex = W.ToggleButtons(options=[('ชาย', 'male'), ('หญิง', 'female')], value='female', description='เพศ', style=style)
sbp = W.FloatText(value=113, description='ความดันตัวบน', style=style)
dbp = W.FloatText(value=71, description='ความดันตัวล่าง', style=style)
weight = W.FloatText(value=67, description='น้ำหนัก', style=style)
unit = W.ToggleButtons(options=['kg', 'lb'], value='kg', style={'button_width': '50px'})
no_weight = W.Checkbox(value=False, description='ไม่ทราบน้ำหนัก', indent=False)
smoking = W.Dropdown(options=[(v, k) for k, v in SMOKING_TH.items()], value='never', description='การสูบบุหรี่', style=style)
diagnosed = W.Checkbox(value=False, description='เคยได้รับการวินิจฉัยว่าเป็นความดันสูงแล้ว', indent=False)
go = W.Button(description='ทำนาย', button_style='primary', icon='heartbeat')
presets = [W.Button(description=f'ตัวอย่าง: {k}') for k in PRESETS]
out = W.Output()

def fill(name):
    a = PRESETS[name]
    age.value, sex.value, sbp.value, dbp.value = a.age, a.sex, a.sbp, a.dbp
    weight.value, unit.value, no_weight.value = a.weight, a.weight_unit, False
    smoking.value, diagnosed.value = a.smoking, False
    run()

def bar(score):
    pct, cut = min(score, 0.5) / 0.5 * 100, THRESHOLD / 0.5 * 100
    color = '#c0392b' if score >= THRESHOLD else '#16a085'
    return f'''<div style="position:relative;height:18px;background:#eee;border-radius:9px;width:420px;margin:6px 0">
      <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:9px"></div>
      <div style="position:absolute;left:{cut:.1f}%;top:-4px;height:26px;border-left:2px dashed #333"></div></div>
      <div style="width:420px;font-size:12px;color:#666">0% ············ เส้นประ = threshold {THRESHOLD:.1%} ············ 50%+</div>'''

def run(_=None):
    out.clear_output()
    answers = Answers(age=age.value, sex=sex.value, sbp=sbp.value, dbp=dbp.value,
                      weight=None if no_weight.value else weight.value, weight_unit=unit.value,
                      smoking=smoking.value, previously_diagnosed=diagnosed.value)
    with out:
        try:
            r = predict(answers, model, state)
        except InputRejected as e:
            display(HTML(f'<div style="padding:10px;border-left:5px solid #e67e22;background:#fdf2e9">'
                         f'<b>ระบบไม่ทำนาย</b><br>{e}</div>'))
            return
        verdict = ('<b style="color:#c0392b">เกิน threshold → แจ้งเตือน</b>' if r.alert
                   else '<b style="color:#16a085">ต่ำกว่า threshold → ไม่แจ้งเตือน</b>')
        rows = ''.join(f'<tr><td>{FEATURE_TH[k]}</td><td style="text-align:right">{v:+.2f}</td></tr>'
                       for k, v in sorted(r.contributions.items(), key=lambda kv: -abs(kv[1])) if abs(v) >= 0.005)
        warn = ''.join(f'<div style="color:#b9770e">⚠ {w}</div>' for w in r.warnings)
        display(HTML(f'''<div style="font-size:15px">คะแนนทดลอง 8 ปี: <b style="font-size:22px">{r.score:.1%}</b>
          &nbsp;(ค่าเฉลี่ยของกลุ่มข้อมูล 6.1%)</div>{bar(r.score)}<div>{verdict}</div>{warn}
          <details style="margin-top:8px"><summary>ทำไมได้คะแนนนี้ (log-odds เทียบคนเฉลี่ยในข้อมูล)</summary>
          <table>{rows}</table><div style="font-size:12px;color:#666">บวก = ดันคะแนนขึ้น · ลบ = ดึงลง</div></details>
          <div style="font-size:12px;color:#888;margin-top:8px">ข้อมูล NHEFS สหรัฐฯ 1982–1992 · label คือการรายงานการวินิจฉัย ·
          ไม่ใช่การวินิจฉัยหรือคำแนะนำทางการแพทย์</div>'''))

go.on_click(run)
for b, name in zip(presets, PRESETS):
    b.on_click(lambda _, n=name: fill(n))

display(W.VBox([W.HBox(presets), age, sex, sbp, dbp, W.HBox([weight, unit, no_weight]),
                smoking, diagnosed, go, out]))"""),
    md("""## ทดสอบโดยไม่ใช้ฟอร์ม (เผื่อ widget ไม่แสดง)

เซลล์นี้รันตัวอย่าง 3 ระดับ และกรณีที่ระบบต้องปฏิเสธ — ใช้ตอน demo หัวข้อ Guardrails ได้"""),
    code("""for name, a in PRESETS.items():
    r = predict(a, model, state)
    print(f'{name:>5}: {r.score:6.2%}  แจ้งเตือน={r.alert}')

print()
cases = {
    'เคยวินิจฉัยแล้ว': Answers(50, 'male', 120, 75, 70, previously_diagnosed=True),
    'ความดัน 135/85': Answers(50, 'male', 135, 85, 70),
    'อายุ 20 ปี': Answers(20, 'male', 115, 70, 70),
    'น้ำหนัก 200 kg': Answers(50, 'male', 115, 70, 200),
    'กรอกความดันสลับ': Answers(50, 'male', 70, 115, 70),
}
for name, a in cases.items():
    try:
        predict(a, model, state); print(f'{name}: ผ่าน (ไม่ควรเกิด)')
    except InputRejected as e:
        print(f'{name}: ปฏิเสธ → {e}')"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
    'language_info': {'name': 'python'}})
Path('LEPTOLIC_Demo.ipynb').write_text(nbf.writes(nb))
print('wrote LEPTOLIC_Demo.ipynb')
