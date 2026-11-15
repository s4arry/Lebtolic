# Codebook ที่ตรวจสำหรับ balancing / แผนสำรอง

อ่านจาก PDF ทางการ CDC ที่ดาวน์โหลดจริง 11 พฤศจิกายน 2026 และ SAS layout ที่มี
SHA-256 ตรงกับไฟล์รอบก่อน หน้า PDF นับจาก 1; หน้าเอกสารที่พิมพ์อาจต่างกัน
การตรวจนี้จำกัดเฉพาะรายการด้านล่าง ไม่ใช่การรับรอง skip logic ทั้งฐานข้อมูล

## ตัวแปรที่อนุญาตให้เพิ่มในการทดลอง

| ตัวแปร | หลักฐาน | การเข้ารหัสในการทดลอง |
|---|---|---|
| PMR5b1, PMR5b2 | [1982 PDF หน้า 372 / หน้าเอกสาร 387](https://wwwn.cdc.gov/nchs/data/nhefs/1982intv.pdf#page=372); SAS columns 4150–4152, 4153–4155 | น้ำหนักที่วัด baseline หน่วย pounds; เฉลี่ย readings ที่มีรหัส 069–300; ถ้ามีเพียงหนึ่งค่าที่ใช้ได้ ใช้ค่านั้น |
| Weight special codes | หน้าเดียวกัน | 995 = ≥301 pounds (top-coded), 996 = unsatisfactory, 997 = inapplicable/มีเพียงครั้งเดียวสำหรับ reading 2, 999 = not ascertained; ไม่ใช้ตัวเลขเหล่านี้เป็น pounds; ถ้าไม่มี reading ใช้ได้ ให้ missing + indicator และ median จาก training fold |
| L1 | [1982 PDF หน้า 242 / หน้าเอกสาร 247](https://wwwn.cdc.gov/nchs/data/nhefs/1982intv.pdf#page=242); SAS column 1991 | เคยสูบ ≥100 มวนตลอดชีวิต: 1 yes, 2 no, 8 unknown, 9 not ascertained |
| L2a | หน้าเดียวกัน; SAS column 1992 | สูบปัจจุบัน: 1 yes, 2 no, 7 inapplicable จาก L1 หรือ proxy; 8/9 unknown |
| Smoking category | กฎผสมจาก L1/L2a ข้างต้น | L1=2 และ L2a=7/ว่าง → never-100; L1=1,L2a=1 → current; L1=1,L2a=2 → former; ที่เหลือ unknown รวม contradiction; ใช้ current/former/unknown indicators มี never-100 เป็น reference |

Weight missingness ไม่ใช่ missing at random ที่ยืนยันแล้ว โดยเฉพาะ top-coded weight
จึงรายงานจำนวนข้อมูลหาย ไม่คาดเดาน้ำหนัก ≥301 ว่าเท่ากับ 301 และไม่เรียกน้ำหนักว่า BMI
ไม่ได้ตรวจยืนยันส่วนสูงที่วัดใน baseline wave นี้ จึงไม่สร้าง BMI จากข้อมูลคนละเวลา
SAS label ทั้งสอง reading เป็น weight; PDF บรรทัดช่วง reading 2 มีคำว่า “Height in pounds”
ที่ไม่สอดคล้องกับหัวข้อ “2nd weight reading” และ label จึงใช้หน่วย pounds ตามหัวข้อหลัก

## การตรวจนิยาม target / follow-up

- [1982 D2 หน้าเอกสาร 53](https://wwwn.cdc.gov/nchs/data/nhefs/1982intv.pdf):
  “Have you ever been told by a doctor ... hypertension or high blood pressure?”
  code 2 เป็นการตอบ no ณ baseline ไม่ใช่หลักฐานทางคลินิกว่าไม่เคยมีโรค
- [1986 BASERSLT PDF หน้า 62 / หน้าเอกสาร 57](https://wwwn.cdc.gov/nchs/data/nhefs/1986intv.pdf#page=62)
  และ [1987 PDF หน้า 31 / หน้าเอกสาร 27](https://wwwn.cdc.gov/nchs/data/nhefs/1987intv.pdf#page=31):
  1 completed subject, 2 completed proxy incapacitated, 3 completed proxy deceased;
  4–6 partial interviews
- [1992 MAINRSLT PDF หน้า 55 / หน้าเอกสาร 52](https://wwwn.cdc.gov/nchs/data/nhefs/1992intv.pdf#page=55):
  C1 subject, C2 proxy incapacitated, C3 proxy deceased; P1–P3 partial
  การระบุ no ตลอด horizon ในโค้ดต้อง C1/C2 พร้อม explicit no และระยะติดตามพอ
- [1992 LHIGHBP PDF หน้า 35 / หน้าเอกสาร 32](https://wwwn.cdc.gov/nchs/data/nhefs/1992intv.pdf#page=35):
  code 1 = รายงานจากรอบก่อน; ใช้กำหนดว่าจะถาม HYPER หรือไม่ จึงห้ามแปลง skip เป็น no
- [1992 HYPER/HBPYEAR1 PDF หน้า 116 / หน้าเอกสาร 113](https://wwwn.cdc.gov/nchs/data/nhefs/1992intv.pdf#page=116):
  HYPER 1 yes, 2 no, 95 borderline, 96 discrepancy; -1 skip; -7/-8/-9 refused/unknown/missing;
  ปี 20–93 หมายถึง 1920–1993 ไม่ใช่อายุ
- [1992 NEWDODYR PDF หน้า 66 / หน้าเอกสาร 63](https://wwwn.cdc.gov/nchs/data/nhefs/1992intv.pdf#page=66):
  year 78–93, -1 inapplicable ยืนยันว่าเป็นปีตายที่โค้ด guard ใช้ได้สำหรับ wave นี้
  ปีตายก่อน/ทับปี horizon ถูกคัด negative ออกอย่างอนุรักษ์นิยม

รอบนี้รักษา target และ cohort เดิมเพื่อให้เทียบ balancing ได้ ยืนยันความหมายรหัสที่อ่าน
ไม่ได้แปลว่าวันเริ่มโรค/ปีที่จำย้อนหลังถูกต้อง หรือแก้ selection bias/competing risk แล้ว
การเทียบ included กับ excluded แยกผู้ที่ไม่ผ่าน baseline ออกจากผู้ผ่าน baseline แต่
follow-up ไม่พอ/ขัดแย้ง เพื่อไม่ปะปนสาเหตุการคัดออก การไม่ติดตามต่อไม่ใช่ negative
