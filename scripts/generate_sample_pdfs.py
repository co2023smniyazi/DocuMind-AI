import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

docs_data = [
    {
        "filename": "academic_regulations.pdf",
        "title": "Academic Regulations and Curriculum Guidelines",
        "sections": [
            (
                "1. Minimum Attendance Requirements",
                "A student must maintain a minimum attendance of 75% in the aggregate of all courses (theory and practicals) in a semester to be eligible to appear for the semester end examinations. "
                "Condonation of shortage of attendance between 65% and 74% may be granted on medical grounds or genuine valid reasons by the Academic Committee upon payment of a prescribed condonation fee of Rs 500. "
                "Students with attendance below 65% in aggregate are not eligible for condonation under any circumstances and will be detained for that semester and must repeat the semester in the subsequent academic year."
            ),
            (
                "2. Credit System and Graduation Requirements",
                "For the award of the B.Tech Degree (4-year program), a candidate is required to successfully earn a minimum of 160 credits across 8 semesters. "
                "For students admitted under the Lateral Entry scheme into the second year (3-year program), a minimum of 120 credits is required for graduation. "
                "A student must secure a minimum Cumulative Grade Point Average (CGPA) of 5.0 on a 10-point scale with no active backlogs to be awarded the degree. "
                "Award of First Class with Distinction requires a CGPA of 7.5 and above in the first attempt. First Class requires a CGPA of 6.5 to 7.49."
            ),
            (
                "3. Grading System and Evaluation Scale",
                "Academic performance is graded on a 10-point scale: "
                "Grade 'O' (Outstanding) corresponds to 10 points (>=90% marks); "
                "Grade 'A+' (Excellent) corresponds to 9 points (80-89% marks); "
                "Grade 'A' (Very Good) corresponds to 8 points (70-79% marks); "
                "Grade 'B+' (Good) corresponds to 7 points (60-69% marks); "
                "Grade 'B' (Above Average) corresponds to 6 points (50-59% marks); "
                "Grade 'C' (Pass) corresponds to 5 points (40-49% marks); "
                "Grade 'F' (Fail) corresponds to 0 points (<40% marks); "
                "Grade 'Ab' indicates Absent."
            ),
            (
                "4. Promotion and Detention Rules",
                "A student shall be promoted from the 1st year to the 2nd year only if they secure at least 50% of the total credits prescribed for the 1st year. "
                "Promotion from the 2nd year to the 3rd year requires securing at least 60% of total credits prescribed up to the 4th semester. "
                "Promotion from the 3rd year to the 4th year requires securing at least 60% of the total credits prescribed up to the 6th semester. "
                "Students who fail to meet the credit requirement will be detained in the respective year."
            ),
            (
                "5. Semester Registration and Required Documents",
                "Every student must register for each semester within 7 days of the reopening date. "
                "The documents required for semester registration are: "
                "(a) Copy of previous semester fee payment receipt, "
                "(b) Grade sheet / marks memo of the preceding semester, "
                "(c) Signed Anti-Ragging undertaking affidavit by student and parent, "
                "(d) Faculty advisor course clearance approval form, "
                "(e) Hostel clearance certificate if applicable."
            )
        ]
    },
    {
        "filename": "examination_guidelines.pdf",
        "title": "Examination Rules, Evaluation and Revaluation Guidelines",
        "sections": [
            (
                "1. Hall Ticket and Examination Eligibility",
                "Students are eligible to obtain the examination hall ticket only after clearing all institutional dues and securing the minimum prescribed attendance of 75%. "
                "Hall tickets must be presented along with the college identity card at every examination session. "
                "No candidate will be allowed to enter the examination hall after 15 minutes from the scheduled commencement of the test."
            ),
            (
                "2. Procedure for Examination Revaluation",
                "A student who desires to apply for revaluation of an answer script can submit an application through the student examination portal within 10 days from the date of declaration of results. "
                "The revaluation fee is Rs. 750 per subject course. "
                "If the difference between original marks and revalued marks is 15% or more, the revised marks are awarded. "
                "If revaluation results in an increase, the higher mark will be considered for the final grade."
            ),
            (
                "3. Backlog and Supplementary Examination Rules",
                "Supplementary examinations for backlog courses are conducted twice an academic year. "
                "Odd-semester backlogs can be cleared during odd-semester supplementary sessions, and even-semester backlogs during even-semester sessions. "
                "A student is permitted a maximum of 3 consecutive attempts to clear any backlog subject. "
                "If a student fails to clear within 3 attempts, special remedial mentoring and departmental permission is required for re-enrolment."
            ),
            (
                "4. Malpractice and Unfair Means Penalties",
                "Possession of mobile phones, smartwatches, programmable calculators, or unauthorized study material inside the examination hall is treated as malpractice. "
                "First offense: Cancellation of performance in that specific subject and award of Grade F. "
                "Second offense or copying from materials: Cancellation of all examinations of that semester and debarment from the college for one subsequent semester."
            ),
            (
                "5. Challenge Valuation Scheme",
                "If a candidate is dissatisfied with the revaluation outcome, they may request a Challenge Valuation within 7 days of revaluation results. "
                "The challenge valuation fee is Rs. 3,000 per subject. "
                "If the marks increase by 20% or more upon evaluation by an external senior examiner, the entire fee of Rs. 3,000 is refunded to the student."
            )
        ]
    },
    {
        "filename": "student_handbook.pdf",
        "title": "Student Code of Conduct and Campus Handbook",
        "sections": [
            (
                "1. Campus Timings and Working Hours",
                "The college operates from 9:00 AM to 4:30 PM, Monday through Friday. "
                "Second and fourth Saturdays are recognized holidays, while first and third Saturdays are active instruction days for labs and tutorials. "
                "Students residing in the hostel must return to hostel premises by 8:00 PM on weekdays and 9:00 PM on weekends."
            ),
            (
                "2. Anti-Ragging Policy and Strict Prohibition",
                "Ragging is strictly prohibited anywhere on campus, in hostels, and in transit vehicles under the Supreme Court directives and UGC Regulations. "
                "Ragging is a non-bailable criminal offense. Any student found guilty faces immediate suspension, registration of police FIR, and expulsion. "
                "Incidents can be reported 24/7 to the College Anti-Ragging Cell or the National Anti-Ragging Toll-Free Helpline at 1800-180-5522."
            ),
            (
                "3. Student Identity Card and Dress Code",
                "Every student must wear their RFID Identity Card visibly around their neck while on campus premises. "
                "Formal dress code is mandatory on Mondays, Tuesdays, and Thursdays: collared shirts and formal trousers for boys; formal salwar/kurtas or trousers for girls. "
                "Smart casual attire is allowed on Wednesdays and Fridays. T-shirts with offensive slogans and slippers are strictly forbidden."
            ),
            (
                "4. Library Rules and Book Borrowing Regulations",
                "Undergraduate students are eligible to borrow up to 4 books simultaneously for a maximum borrowing period of 14 days. "
                "Books can be renewed once if there are no pending reservations. "
                "An overdue fine of Rs. 2 per book per day will be levied for late returns. "
                "The Digital Library and Reference Section are open from 8:00 AM to 7:00 PM on all working days."
            ),
            (
                "5. Student Grievance Redressal Mechanism",
                "Any academic or non-academic grievance can be lodged through the Student ERP portal or submitted directly in writing to the Dean of Student Affairs. "
                "The Grievance Committee investigates all registered complaints and issues a formal resolution within 7 working days."
            )
        ]
    },
    {
        "filename": "placement_guidelines.pdf",
        "title": "Training and Placement Cell Policy and Guidelines",
        "sections": [
            (
                "1. Placement Eligibility Criteria",
                "Students eligible for campus recruitment drives must maintain an aggregate of at least 60% or a minimum CGPA of 6.5 throughout their academic career without any active standing backlogs. "
                "Certain tier-1 recruiters require a minimum CGPA of 7.5 or 8.0 with no history of backlogs. "
                "Final-year students who fulfill these academic thresholds must register on the college placement portal by August 15th."
            ),
            (
                "2. One-Student-One-Job Policy",
                "The Training and Placement Cell follows a strict 'One-Student-One-Job' policy to ensure equitable employment opportunities. "
                "Once a candidate receives a formal job offer, they will be marked as 'Placed' and are ineligible for subsequent recruitment drives. "
                "Exception: A placed student is permitted to participate in 'Dream Company' drives only if the Dream offer CTC is at least 2 times the current offer CTC."
            ),
            (
                "3. Training and Mock Interview Attendance Requirements",
                "To remain eligible for campus placement drives, students must maintain a mandatory minimum attendance of 85% in all soft skills, technical training workshops, aptitude bootcamps, and mock interviews. "
                "Unexcused absence from two consecutive training sessions will result in placement registration suspension for a duration of one month."
            ),
            (
                "4. Placement Dress Code and Interview Etiquette",
                "Students must appear in full professional formal attire (formal suit/blazer, white shirt, formal shoes, tie for boys) for both on-campus and virtual interviews. "
                "Students must carry 3 hard copies of their approved resume and their college ID card. "
                "Any candidate who behaves unprofessionally or fails to appear after shortlisting will be permanently debarred from further placement assistance."
            ),
            (
                "5. Offer Acceptance and Reporting Deadlines",
                "Upon receiving an offer letter from a participating company, the student must formally accept or decline the offer within 48 hours. "
                "If a candidate accepts an offer and subsequently refuses to join without valid medical justification, their degree certificate recommendation will be withheld."
            )
        ]
    },
    {
        "filename": "department_rules.pdf",
        "title": "Department Rules, Laboratory Safety and Mini-Project Guidelines",
        "sections": [
            (
                "1. Laboratory Attendance and Conduct",
                "Students must attend 100% of all scheduled laboratory practicals. "
                "If a laboratory session is missed due to verified medical illness, the student must attend the designated Saturday makeup laboratory session within 10 days. "
                "Laboratory observation notebooks and completed experiment records must be verified and signed by the faculty-in-charge in the very next lab session."
            ),
            (
                "2. Laboratory Safety Precautions",
                "Safety is the highest priority in departmental laboratories. "
                "Wearing closed-toe shoes and cotton lab coats is mandatory in electrical, mechanical, and hardware laboratories. "
                "Consuming food, beverages, or water near computer terminals is strictly prohibited. "
                "Connecting unverified personal external flash drives to laboratory systems is forbidden to prevent malware contamination."
            ),
            (
                "3. Mini-Project Guidelines and Milestone Schedule",
                "The final-year mini project carries 2 credits and must be executed in teams of 3 to 4 students. "
                "Project milestones must strictly be adhered to: "
                "Milestone 1 (Week 4): Project abstract, problem definition, and Jira setup approval; "
                "Milestone 2 (Week 8): Mid-semester progress demo, RAG pipeline and vector store implementation; "
                "Milestone 3 (Week 14): Final working demonstration, Streamlit UI, evaluation report, and documentation submission."
            ),
            (
                "4. Mini-Project Evaluation Rubric",
                "The 100-mark mini-project evaluation is broken down as follows: "
                "(a) Problem Definition and Functional Requirements: 15 marks; "
                "(b) System Architecture and RAG Implementation: 30 marks; "
                "(c) Jira Project Management and Sprint Tracking: 20 marks; "
                "(d) Functional Testing, Retrieval Evaluation and Hallucination Assessment: 15 marks; "
                "(e) Final Presentation, Viva Voce and Demonstration: 20 marks."
            ),
            (
                "5. Plagiarism and Academic Integrity Policy",
                "All project reports, documentation, and technical code submissions will be scanned using institutional plagiarism detection software. "
                "The report must have an overall similarity index of less than 15%, excluding code snippets and bibliography references. "
                "Submissions with plagiarism exceeding 25% will be summarily rejected and the team must re-register for the course."
            )
        ]
    }
]

def generate_pdf(doc_info):
    file_path = os.path.join(DATA_DIR, doc_info["filename"])
    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1E40AF'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceAfter=8
    )
    
    story = []
    story.append(Paragraph(doc_info["title"], title_style))
    story.append(Paragraph("<b>Institution:</b> College of Engineering & Technology | <b>Official Policy Document</b>", body_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3B82F6'), spaceAfter=14))
    
    for heading, text in doc_info["sections"]:
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(text, body_style))
        story.append(Spacer(1, 6))
        
    doc.build(story)
    print(f"Generated: {file_path}")

if __name__ == "__main__":
    for doc in docs_data:
        generate_pdf(doc)
    print("All 5 college PDF documents successfully created in data/ directory!")

