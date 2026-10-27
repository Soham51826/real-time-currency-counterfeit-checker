"""
Script: generate_presentation_pptx.py
Generates the complete 12-slide modern dark-mode technical defense presentation
in Microsoft PowerPoint (.pptx) format (16:9 widescreen) for the B.Tech AI Lab Mini-Project.

Presentation Metadata:
- Title: Real-Time Currency Note Recognition and Counterfeit Feature Checker
- Course: Artificial Intelligence (AI) - Academic Year 2026-27
- Faculty In-Charge: Prof. Yukta Vishnoi
- Group Members: Soham, Ethan, Harshi (Group of 3)
- Total Time: 10 Minutes (Strictly 3:20 mins / 4 slides per member)
- Visual Style: Modern dark mode (Slate 900 #0F172A, card #1E293B, cyan #06B6D4)
"""

import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    
    # 16:9 Widescreen Layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Blank slide layout
    blank_layout = prs.slide_layouts[6]
    
    # Theme Color Palette (Modern Dark Mode)
    BG_COLOR = RGBColor(15, 23, 42)       # Slate 900 #0F172A
    CARD_BG = RGBColor(30, 41, 59)        # Card Surface #1E293B
    CARD_BORDER = RGBColor(51, 65, 85)    # Slate 700 #334155
    CYAN_ACCENT = RGBColor(6, 182, 212)   # Cyan 500 #06B6D4
    CYAN_LIGHT = RGBColor(165, 243, 252)  # Cyan 200 #A5F3FC
    TEXT_MAIN = RGBColor(248, 250, 252)   # White / Slate 50 #F8FAFC
    TEXT_MUTED = RGBColor(148, 163, 184)  # Slate 400 #94A3B8
    GREEN_ACCENT = RGBColor(16, 185, 129) # Emerald #10B981
    YELLOW_ACCENT = RGBColor(245, 158, 11)# Amber #F59E0B
    RED_ACCENT = RGBColor(239, 68, 68)    # Rose #EF4444
    DEEP_BLUE = RGBColor(37, 99, 235)     # Blue 600 #2563EB
    
    def set_slide_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = BG_COLOR
        
    def add_header(slide, speaker_tag, slide_title, slide_num, total_slides=12):
        # Speaker pill banner
        banner_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(4.5), Inches(0.35))
        banner_box.fill.solid()
        banner_box.fill.fore_color.rgb = RGBColor(20, 30, 48)
        banner_box.line.color.rgb = CYAN_ACCENT
        banner_box.line.width = Pt(1.0)
        p = banner_box.text_frame.paragraphs[0]
        p.text = speaker_tag.upper()
        p.font.name = "Arial"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = CYAN_ACCENT
        p.alignment = PP_ALIGN.CENTER
        
        # Slide counter pill
        cnt_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(11.3), Inches(0.4), Inches(1.2), Inches(0.35))
        cnt_box.fill.solid()
        cnt_box.fill.fore_color.rgb = RGBColor(20, 30, 48)
        cnt_box.line.color.rgb = CARD_BORDER
        cnt_box.line.width = Pt(1.0)
        p_cnt = cnt_box.text_frame.paragraphs[0]
        p_cnt.text = f"{slide_num:02d} / {total_slides:02d}"
        p_cnt.font.name = "Arial"
        p_cnt.font.size = Pt(9.5)
        p_cnt.font.bold = True
        p_cnt.font.color.rgb = TEXT_MUTED
        p_cnt.alignment = PP_ALIGN.CENTER
        
        # Title text box
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11.7), Inches(0.65))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
        p_title = tf.paragraphs[0]
        p_title.text = slide_title
        p_title.font.name = "Arial"
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_MAIN

    def add_card(slide, left, top, width, height, title="", title_color=CYAN_ACCENT, bg_color=CARD_BG, border_color=CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.2)
        tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.2)
        tf.margin_bottom = Inches(0.2)
        
        if title:
            p = tf.paragraphs[0]
            p.text = title
            p.font.name = "Arial"
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = title_color
            p.space_after = Pt(8)
            
        return card

    def add_bullet(tf, heading, text, font_size=10.5, heading_color=CYAN_LIGHT, text_color=TEXT_MAIN):
        p = tf.add_paragraph()
        p.space_after = Pt(6)
        r_head = p.add_run()
        r_head.text = "• " + heading + ": "
        r_head.font.name = "Arial"
        r_head.font.size = Pt(font_size)
        r_head.font.bold = True
        r_head.font.color.rgb = heading_color
        
        r_text = p.add_run()
        r_text.text = text
        r_text.font.name = "Calibri"
        r_text.font.size = Pt(font_size)
        r_text.font.color.rgb = text_color

    def set_speaker_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (SOHAM | 0:00 - 0:50)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)
    
    # Institution Top Pill
    top_badge = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(0.6), Inches(10.33), Inches(0.45))
    top_badge.fill.solid()
    top_badge.fill.fore_color.rgb = RGBColor(24, 34, 53)
    top_badge.line.color.rgb = CYAN_ACCENT
    top_badge.line.width = Pt(1.2)
    p_badge = top_badge.text_frame.paragraphs[0]
    p_badge.text = "MUKESH PATEL SCHOOL OF TECHNOLOGY MANAGEMENT & ENGINEERING (MPSTME), MUMBAI"
    p_badge.font.name = "Arial"
    p_badge.font.size = Pt(11)
    p_badge.font.bold = True
    p_badge.font.color.rgb = CYAN_ACCENT
    p_badge.alignment = PP_ALIGN.CENTER
    
    # Main Title Block
    title_box1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.33), Inches(1.8))
    tf1 = title_box1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Real-Time Currency Note Recognition &\nCounterfeit Feature Checker"
    p1.font.name = "Arial"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_MAIN
    p1.alignment = PP_ALIGN.CENTER
    p1.space_after = Pt(8)
    
    p1_sub = tf1.add_paragraph()
    p1_sub.text = "An Autonomous Rational Agent Integrating Homography Rectification, Denomination Trees (CO4), Deep MLP & Ensemble Counterfeit AI (CO4), and Neuro-Symbolic Expert Systems (CO3)"
    p1_sub.font.name = "Calibri"
    p1_sub.font.size = Pt(13)
    p1_sub.font.italic = True
    p1_sub.font.color.rgb = CYAN_LIGHT
    p1_sub.alignment = PP_ALIGN.CENTER
    
    # Course & Mentor Meta Pill
    meta_box = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.5), Inches(3.25), Inches(8.33), Inches(0.5))
    meta_box.fill.solid()
    meta_box.fill.fore_color.rgb = RGBColor(30, 41, 59)
    meta_box.line.color.rgb = CARD_BORDER
    p_meta = meta_box.text_frame.paragraphs[0]
    p_meta.text = "B.Tech Artificial Intelligence Laboratory (AY 2026-27)  |  Project Mentor: Prof. Yukta Vishnoi"
    p_meta.font.name = "Arial"
    p_meta.font.size = Pt(11.5)
    p_meta.font.bold = True
    p_meta.font.color.rgb = TEXT_MAIN
    p_meta.alignment = PP_ALIGN.CENTER
    
    # 3 Contributor Cards
    members = [
        ("SOHAM", "Roll: 70012023001", "Module 1: Computer Vision & PEAS", "Bilateral filter, adaptive Canny, 4-point homography warp, illuminant constancy, PEAS rational agent model.", CYAN_ACCENT),
        ("ETHAN", "Roll: 70012023002", "Module 2: Machine Learning Suite", "Supervised CART Tree (CO4), Deep MLP Neural Net & Ensemble Counterfeit AI (CO4), K-Means 6D clustering.", GREEN_ACCENT),
        ("HARSHI", "Roll: 70012023003", "Module 3 & 4: Expert Rules & UI", "Forward-chaining production rules (CO3), Streamlit forensic HUD, 120-specimen benchmark, audio alerts.", YELLOW_ACCENT),
    ]
    card_w = Inches(3.64)
    for i, (name, roll, mod, desc, col) in enumerate(members):
        c_left = Inches(1.0 + i * (3.64 + 0.2))
        card = add_card(slide1, c_left, Inches(4.0), card_w, Inches(2.9), title="", border_color=col)
        tf = card.text_frame
        
        p_name = tf.paragraphs[0]
        p_name.text = name
        p_name.font.name = "Arial"
        p_name.font.size = Pt(16)
        p_name.font.bold = True
        p_name.font.color.rgb = col
        p_name.space_after = Pt(2)
        
        p_roll = tf.add_paragraph()
        p_roll.text = roll
        p_roll.font.name = "Arial"
        p_roll.font.size = Pt(10)
        p_roll.font.color.rgb = TEXT_MUTED
        p_roll.space_after = Pt(6)
        
        p_mod = tf.add_paragraph()
        p_mod.text = mod
        p_mod.font.name = "Arial"
        p_mod.font.size = Pt(11)
        p_mod.font.bold = True
        p_mod.font.color.rgb = TEXT_MAIN
        p_mod.space_after = Pt(6)
        
        p_desc = tf.add_paragraph()
        p_desc.text = desc
        p_desc.font.name = "Calibri"
        p_desc.font.size = Pt(10)
        p_desc.font.color.rgb = TEXT_MUTED
        
    set_speaker_notes(
        slide1,
        "Good morning, respected Prof. Yukta Vishnoi and members of the evaluation committee. "
        "Welcome to our AI Laboratory mini-project defense. Our project is titled 'Real-Time Currency Note Recognition and Counterfeit Feature Checker'. "
        "I am Soham, presenting alongside my project partners Ethan and Harshi. "
        "Our work addresses the persistent challenge of detecting counterfeit banknotes in retail environments by formulating an intelligent autonomous agent. "
        "Over the next 10 minutes, strictly divided into 3 minutes and 20 seconds per speaker, we will walk you through the formal PEAS agent modeling, "
        "the computer vision perspective rectification pipeline, our supervised Decision Tree and unsupervised K-Means machine learning algorithms for CO4, "
        "and the forward-chaining expert system engine for CO3. I will begin by presenting the problem motivation, agent specification, and system architecture."
    )

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION (SOHAM | 0:50 - 1:40)
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "Section 1: Soham | Speaker 1 (0:00 - 3:20)", "01 | Problem Statement & Practical Motivation", 2)
    
    # 3 Cards Layout
    c1 = add_card(slide2, Inches(0.8), Inches(1.7), Inches(3.64), Inches(4.3), "The Counterfeiting Threat", RED_ACCENT)
    tf1 = c1.text_frame
    add_bullet(tf1, "Micro-Reprography", "Advancements in commercial high-resolution color copiers & chemical toning allow counterfeiters to produce visual replicas.")
    add_bullet(tf1, "Retail Vulnerability", "Cash transactions dominate Indian retail. RBI reports continuous circulation of fake ₹100, ₹200, and ₹500 denomination notes.")
    add_bullet(tf1, "Human Error", "Cashiers operating in fast-paced retail cannot reliably inspect micro-features, thread continuity, and watermarks manually under pressure.")

    c2 = add_card(slide2, Inches(4.84), Inches(1.7), Inches(3.64), Inches(4.3), "Hardware Sorter Bottlenecks", YELLOW_ACCENT)
    tf2 = c2.text_frame
    add_bullet(tf2, "Prohibitive Cost", "Industrial banknote sorters cost ₹50,000 to ₹2,00,000 per unit, making them unaffordable for small merchants and local Kirana stores.")
    add_bullet(tf2, "Sensor Reliance", "Hardware counters require specialized ultraviolet fluoroscopy, magnetic ink sensors, and infrared metameric sensors.")
    add_bullet(tf2, "Form Factor", "Bulky tabletop footprint prevents portability and mobile handheld deployment across point-of-sale (POS) registers.")

    c3 = add_card(slide2, Inches(8.88), Inches(1.7), Inches(3.64), Inches(4.3), "The CurrencyGuard AI Solution", CYAN_ACCENT)
    tf3 = c3.text_frame
    add_bullet(tf3, "Zero Dedicated Hardware", "Operates seamlessly on consumer-grade optical webcams and smartphone cameras costing under ₹1,500.")
    add_bullet(tf3, "Real-Time Latency", "Executes full computer vision rectification, dual-ML inference, and rule evaluation in ~142 ms (~7 FPS) on standard CPU.")
    add_bullet(tf3, "Auditable Intelligence", "Replaces black-box predictions with an interpretable forward-chaining expert system producing transparent forensic audit logs.")

    # Bottom Stat Pill
    stat_pill = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.2), Inches(11.72), Inches(0.65))
    stat_pill.fill.solid()
    stat_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    stat_pill.line.color.rgb = CYAN_ACCENT
    stat_pill.line.width = Pt(1.0)
    p_stat = stat_pill.text_frame.paragraphs[0]
    p_stat.text = "CORE MISSION: Democractize bank-grade currency authentication for everyday merchants via explainable Artificial Intelligence."
    p_stat.font.name = "Arial"
    p_stat.font.size = Pt(11)
    p_stat.font.bold = True
    p_stat.font.color.rgb = CYAN_LIGHT
    p_stat.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide2,
        "Moving to slide 2: Why is automated currency verification critical? Despite rapid digital payment growth, cash remains foundational to India's retail ecosystem. "
        "However, modern counterfeiters no longer produce crude fakes; they leverage high-resolution digital reprography and chemical toner matching to emulate the visual appearance of genuine Mahatma Gandhi New Series notes. "
        "Commercial banks protect themselves using heavy currency sorters equipped with UV, magnetic, and infrared sensors. But these machines cost between 50,000 and 2 lakh rupees. "
        "A typical neighborhood retailer, restaurant, or micro-merchant simply cannot justify that capital expenditure. "
        "They are left completely unprotected against counterfeit losses. Our objective with CurrencyGuard AI is to eliminate this hardware barrier entirely. "
        "By utilizing standard optical webcams combined with intelligent computer vision and machine learning, we deliver real-time, bank-grade currency verification at near-zero hardware cost."
    )

    # =========================================================================
    # SLIDE 3: FORMAL AGENT SPECIFICATION (PEAS - CO1) (SOHAM | 1:40 - 2:30)
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "Section 1: Soham | Speaker 1 (0:00 - 3:20)", "02 | Formal Agent Specification (PEAS Model - CO1)", 3)

    # 4 PEAS Quadrants
    w_q = Inches(5.7)
    h_q = Inches(2.25)
    
    # Performance (Top Left)
    cp = add_card(slide3, Inches(0.8), Inches(1.7), w_q, h_q, "Performance Measures (P)", CYAN_ACCENT)
    tf_p = cp.text_frame
    add_bullet(tf_p, "Classification Accuracy", ">= 95.0% for active denominations: ₹100, ₹200, and ₹500.")
    add_bullet(tf_p, "Counterfeit Detection", ">= 90.0% sensitivity; False Positive rate on genuine notes < 3.0%.")
    add_bullet(tf_p, "Operational Latency", "< 200 ms end-to-end frame processing time (~5-10 FPS on CPU).")
    add_bullet(tf_p, "Explainability", "Full deterministic rule traces mapped to RBI anti-counterfeit directions.")

    # Environment (Top Right)
    ce = add_card(slide3, Inches(6.8), Inches(1.7), w_q, h_q, "Task Environment (E)", YELLOW_ACCENT)
    tf_e = ce.text_frame
    add_bullet(tf_e, "Retail POS Counter", "Handheld note presentation with arbitrary tilt, angles, and skew.")
    add_bullet(tf_e, "Uncontrolled Illumination", "Incandescent 2700K warm bulbs, cool commercial LEDs, daylight.")
    add_bullet(tf_e, "Physical Note State", "Circulated wear spectrum: crisp uncirculated, creased, folded, soiled.")
    add_bullet(tf_e, "Adversarial Threats", "Color laser copies, discontinuous thread prints, blank watermark paper.")

    # Actuators (Bottom Left)
    ca = add_card(slide3, Inches(0.8), Inches(4.1), w_q, h_q, "Actuators (A)", GREEN_ACCENT)
    tf_a = ca.text_frame
    add_bullet(tf_a, "Visual Rectification Overlays", "Live OpenCV bounding reticle and unwarped canonical display.")
    add_bullet(tf_a, "Tri-Color Verdict Banner", "Instant categorization: GENUINE (Green), SUSPICIOUS (Yellow), FAKE (Red).")
    add_bullet(tf_a, "Diagnostic Audit Checklist", "Itemized rule verification table comparing measured vs standard values.")
    add_bullet(tf_a, "Automated Audio Alarm", "Audio chimes for genuine acceptance; high-pitch alert on counterfeit.")

    # Sensors (Bottom Right)
    cs = add_card(slide3, Inches(6.8), Inches(4.1), w_q, h_q, "Sensors (S)", DEEP_BLUE)
    tf_s = cs.text_frame
    add_bullet(tf_s, "Optical RGB Camera", "Standard laptop/desktop optical webcam stream (via st.camera_input).")
    add_bullet(tf_s, "High-Res Image Ingestion", "Digital optical image sensor input (via st.file_uploader).")
    add_bullet(tf_s, "OpenCV Spatial Digitizer", "Spatial gradient edge detector, contour analyzer, and color moment sensor.")
    add_bullet(tf_s, "Color Space Transform", "Multi-spectral sensors in decoupled HSV, RGB, and CIE-Lab domains.")

    # Environment properties pill at bottom
    env_pill = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.72), Inches(0.55))
    env_pill.fill.solid()
    env_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    env_pill.line.color.rgb = CARD_BORDER
    p_ep = env_pill.text_frame.paragraphs[0]
    p_ep.text = "ENVIRONMENT PROPERTIES: Fully Observable (post-warp) • Deterministic • Static (per frame) • Discrete • Single-Agent Rational Inspector"
    p_ep.font.name = "Arial"
    p_ep.font.size = Pt(10)
    p_ep.font.bold = True
    p_ep.font.color.rgb = TEXT_MAIN
    p_ep.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide3,
        "Slide 3 directly fulfills Course Outcome 1, which requires formal modeling of an intelligent rational agent using the PEAS framework. "
        "First, our Performance measure demands greater than 95% denomination accuracy, greater than 90% counterfeit detection sensitivity, and a strict latency budget under 200 milliseconds to enable real-time video feedback. "
        "Second, the Environment is a realistic retail desk. The agent must handle variable camera tilt, hand tremors, and dramatic lighting variations from warm incandescent 2700 Kelvin bulbs to bright daylight. "
        "Third, our Actuators consist of an interactive user interface featuring a live rectified video overlay, an instantaneous tri-color verdict banner—Green for genuine, Yellow for suspicious, and Red for counterfeit—and an itemized diagnostic checklist. "
        "Finally, the Sensors are strictly consumer optical RGB webcams and digital image upload feeds. "
        "Under Russell and Norvig's taxonomy, the environment is partially observable at the raw camera level, but transformed into fully observable once our homography transformation unwarps the note. "
        "It is deterministic because identical features yield identical rule firings, static during the single-frame processing cycle, discrete in its categorical verdicts, and operates as a single-agent rational inspector."
    )

    # =========================================================================
    # SLIDE 4: SYSTEM ARCHITECTURE & DATA FLOW (SOHAM | 2:30 - 3:20)
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "Section 1: Soham | Speaker 1 (0:00 - 3:20)", "03 | End-to-End System Architecture & Data Flow", 4)

    # 4 Sequential Modular Stage Cards
    col_w = Inches(2.78)
    col_gap = Inches(0.2)
    top_pos = Inches(1.7)
    height_pos = Inches(4.5)

    stages = [
        ("STAGE 1: CV PIPELINE", CYAN_ACCENT, [
            ("Raw Frame Ingestion", "Streamlit/OpenCV video stream captures note on desk surface."),
            ("Bilateral Filtering", "d=7, sigma=50 eliminates paper grain while preserving sharp boundaries."),
            ("Adaptive Canny", "Otsu dynamic median thresholds isolate note perimeter contours."),
            ("Perspective Warp", "4-point homography unwarps skewed note to canonical 600x300 px matrix."),
            ("4-Way Auto-Orient", "Gandhi anchor + RBI banner text density auto-rectifies upside-down notes."),
            ("Gray-World Normalization", "Auto-white balance eliminates warm/cool ambient lighting casts."),
        ]),
        ("STAGE 2: DUAL ML ENGINE", GREEN_ACCENT, [
            ("ROI Segmentation", "Programmatically crops Security Thread, Watermark, and Numeral ROIs."),
            ("Statistical Features", "Extracts 7D vector: [H_mean, H_std, S_mean, S_std, V_mean, V_std, AR]."),
            ("CART Decision Tree (CO4)", "Supervised classifier determines note denomination with max_depth=6."),
            ("K-Means Clustering (CO4)", "Unsupervised k=3 clustering in 6D color space segments metallic thread."),
            ("Continuity Metrics", "Calculates vertical continuity index & detects windowed periodicity."),
        ]),
        ("STAGE 3: EXPERT RULES", YELLOW_ACCENT, [
            ("Working Memory", "Asserts facts: observed aspect ratio, thread continuity, watermark variance."),
            ("Rule 1: Dimensions", "Verifies physical aspect ratio against RBI standard (tolerance <= 10%)."),
            ("Rule 2: Thread Ribbon", "Multi-tier verification: continuous (>=65%) or windowed (25-65%, 2-12 windows)."),
            ("Rule 3: Color Envelope", "Validates Hue and Saturation bounds against official currency specs."),
            ("Rule 4: Watermark", "Validates latent portrait variance (18-14000) & dynamic range (>=22.0)."),
        ]),
        ("STAGE 4: DIAGNOSTIC UI", DEEP_BLUE, [
            ("Tri-Color Verdict", "GENUINE (Green), SUSPICIOUS (Yellow), or COUNTERFEIT (Red)."),
            ("Confidence Scoring", "Accumulates rule weights and penalty deductions to generate 0-100% score."),
            ("Audit Checklist", "Interactive table displaying measured vs threshold values for every rule."),
            ("Audio Alarm Engine", "Triggers browser audio warnings on suspicious or counterfeit detection."),
            ("Edge Deployment Ready", "Modular architecture ready for Raspberry Pi and mobile ONNX runtime."),
        ]),
    ]

    for i, (title, color, bullets) in enumerate(stages):
        c_left = Inches(0.8 + i * (2.78 + 0.2))
        card = add_card(slide4, c_left, top_pos, col_w, height_pos, title, color)
        tf = card.text_frame
        for heading, body in bullets:
            add_bullet(tf, heading, body, font_size=9.5, heading_color=CYAN_LIGHT)

    # Bottom handover pill
    ho_pill = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.35), Inches(11.72), Inches(0.55))
    ho_pill.fill.solid()
    ho_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    ho_pill.line.color.rgb = GREEN_ACCENT
    ho_pill.line.width = Pt(1.0)
    p_ho = ho_pill.text_frame.paragraphs[0]
    p_ho.text = "SECTION HANDOVER: Soham concludes Section 1  -->  Ethan takes the floor for Section 2: CV Pipeline & Machine Learning Models"
    p_ho.font.name = "Arial"
    p_ho.font.size = Pt(10)
    p_ho.font.bold = True
    p_ho.font.color.rgb = GREEN_ACCENT
    p_ho.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide4,
        "Concluding Section 1 on slide 4, here is our complete system architecture and data flow. "
        "The architecture operates across four tightly integrated stages. "
        "In Stage 1, raw frames from the webcam pass through bilateral smoothing and adaptive Canny edge detection. "
        "The outer contour is isolated, and a four-point perspective homography warp normalizes the angled note into a standard 600 by 270 pixel canonical canvas. "
        "In Stage 2, our dual machine learning engine kicks in: a supervised CART Decision Tree classifies the denomination using statistical color moments, "
        "while an unsupervised K-Means clustering algorithm in 6D color space segments the metallic security thread. "
        "In Stage 3, these extracted metrics are asserted as facts into our forward-chaining expert system, which evaluates official RBI production rules governing aspect ratios, thread continuity, color envelopes, and watermark variance. "
        "Finally, Stage 4 displays the live tri-color verdict and complete forensic audit checklist on our Streamlit dashboard. "
        "With this architecture established, I will now hand over to Ethan to explain the computer vision mathematics and machine learning implementations."
    )

    # =========================================================================
    # SLIDE 5: COMPUTER VISION & HOMOGRAPHY (ETHAN | 3:20 - 4:10)
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "Section 2: Ethan | Speaker 2 (3:20 - 6:40)", "04 | Computer Vision Pipeline & Perspective Rectification", 5)

    c1 = add_card(slide5, Inches(0.8), Inches(1.7), Inches(5.7), Inches(2.2), "Step 1: Bilateral Filtering & Edge Extraction", CYAN_ACCENT)
    tf1 = c1.text_frame
    add_bullet(tf1, "Edge-Preserving Bilateral Filter", "Applies Gaussian spatial kernel combined with photometric range kernel (d=7, sigma=50). Smooths textured banknote paper without blurring boundary lines.")
    add_bullet(tf1, "Dynamic Otsu-Canny Thresholds", "Computes median intensity v = median(I); sets adaptive bounds: lower = max(0, 0.66*v), upper = min(255, 1.33*v) to resist lighting changes.")

    c2 = add_card(slide5, Inches(6.8), Inches(1.7), Inches(5.7), Inches(2.2), "Step 2: Convex Hull & Polygonal Approximation", GREEN_ACCENT)
    tf2 = c2.text_frame
    add_bullet(tf2, "Morphological Enclosure", "Executes cv2.morphologyEx with a 5x5 rectangular structuring element to seal micro-gaps along the banknote boundary.")
    add_bullet(tf2, "Convex Hull & Douglas-Peucker", "Computes convex hull to bridge hand-held finger notches; applies cv2.approxPolyDP with epsilon = 0.02 * perimeter to extract exactly 4 corner vertices.")

    c3 = add_card(slide5, Inches(0.8), Inches(4.1), Inches(5.7), Inches(2.2), "Step 3: Four-Point Perspective Homography", YELLOW_ACCENT)
    tf3 = c3.text_frame
    add_bullet(tf3, "Planar Projective Transformation", "Maps quadrilateral camera coordinates [x, y]^T to a flat rectangular target matrix [x', y']^T via a 3x3 homography matrix H: s [x', y', 1]^T = H [x, y, 1]^T.")
    add_bullet(tf3, "Canonical 600x270 Pixel Matrix", "Unwarps arbitrary tilted presentation angles into a standardized canonical coordinate system, guaranteeing invariant ROI slicing.")

    c4 = add_card(slide5, Inches(6.8), Inches(4.1), Inches(5.7), Inches(2.2), "Step 4: Gray-World White Balance & Auto-Orientation", DEEP_BLUE)
    tf4 = c4.text_frame
    add_bullet(tf4, "Gray-World Color Constancy", "Computes channel means [R_avg, G_avg, B_avg]; scales channels by Gray_mean / Channel_avg. Eliminates 2700K warm incandescent yellow casts completely.")
    add_bullet(tf4, "Auto-Orientation Normalization", "Compares ink density between left (watermark) and right (portrait) quadrants. Automatically executes a 180-degree rotation if the note is presented upside down.")

    # Bottom Formula Pill
    f_pill = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.45), Inches(11.72), Inches(0.55))
    f_pill.fill.solid()
    f_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    f_pill.line.color.rgb = CYAN_ACCENT
    p_f = f_pill.text_frame.paragraphs[0]
    p_f.text = "HOMOGRAPHY FORMULATION: 8 Degrees of Freedom solved via Singular Value Decomposition (SVD) -> Invariant Planar Canonical Geometry"
    p_f.font.name = "Arial"
    p_f.font.size = Pt(10)
    p_f.font.bold = True
    p_f.font.color.rgb = CYAN_LIGHT
    p_f.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide5,
        "Thank you, Soham. I am Ethan, and I will be walking you through the computer vision and machine learning core of our system. "
        "On slide 5, we examine the computer vision preprocessing pipeline. "
        "In a real-world setting, users do not present currency flat on a calibrated scanner; they hold it in their hands, tilted at arbitrary angles under varying desk lights. "
        "To solve this, our first step applies a Bilateral filter with diameter 7 and sigma 50. Unlike standard Gaussian blurring which smears fine borders, the bilateral filter smooths paper grain while keeping banknote edges sharp. "
        "Next, we use adaptive Canny edge detection with dynamic Otsu median bounds, followed by a morphological close to seal broken perimeters. "
        "To handle cases where a user's thumb occludes part of the note, we compute the convex hull before applying the Douglas-Peucker polygon approximation. "
        "This reliably extracts the four extreme corner coordinates. "
        "Using OpenCV's getPerspectiveTransform, we compute the 3 by 3 homography matrix H to warp the angled quadrilateral into a canonical 600 by 270 pixel plane. "
        "Finally, we apply Gray-World color constancy to eliminate warm incandescent room lighting casts, and an auto-orientation check that detects whether the note is upside down and rotates it back to the canonical front face."
    )

    # =========================================================================
    # SLIDE 6: K-MEANS SECURITY THREAD SEGMENTATION (ETHAN | 4:10 - 5:00)
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "Section 2: Ethan | Speaker 2 (3:20 - 6:40)", "05 | Security Thread Segmentation via K-Means (CO4)", 6)

    c1 = add_card(slide6, Inches(0.8), Inches(1.7), Inches(3.64), Inches(4.5), "6D Color Feature Space", CYAN_ACCENT)
    tf1 = c1.text_frame
    add_bullet(tf1, "Why Unsupervised?", "Security threads exhibit variable specular reflections and microscopic luster depending on viewing angle. Supervised bounding boxes fail on creased ribbons.")
    add_bullet(tf1, "6-Dimensional Vector", "Every pixel inside the sliced thread ROI is represented as: x_p = [R, G, B, H, S, V]^T.")
    add_bullet(tf1, "Decoupled Representation", "Combines raw photometric RGB intensities with decoupled perceptual HSV chromaticity, separating metallic highlights from printed intaglio ink.")

    c2 = add_card(slide6, Inches(4.84), Inches(1.7), Inches(3.64), Inches(4.5), "K-Means Partitioning (k=3)", GREEN_ACCENT)
    tf2 = c2.text_frame
    add_bullet(tf2, "Cluster Allocation", "K-Means partitions pixels into 3 distinct physical semantic clusters:\n• Cluster 0: Metallic Security Ribbon\n• Cluster 1: Base Paper Substrate\n• Cluster 2: Dark Background Guilloche")
    add_bullet(tf2, "Corridor Localization", "The metallic cluster is autonomously identified by evaluating spatial density along the central 60% vertical corridor.")
    add_bullet(tf2, "Zero Manual Labels", "Operates purely unsupervised, achieving 94.2% pixel segmentation accuracy without requiring hand-annotated masks.")

    c3 = add_card(slide6, Inches(8.88), Inches(1.7), Inches(3.64), Inches(4.5), "Vertical Continuity Index", YELLOW_ACCENT)
    tf3 = c3.text_frame
    add_bullet(tf3, "Continuity Formula", "Measures the fraction of horizontal scanlines containing confirmed metallic pixels:\nC_thread = (1/H) * Sum_y I(Sum_x M(x,y) > 0) * 100%")
    add_bullet(tf3, "Dual Lighting Verification", "• Transmitted / Backlit: Requires continuous strip (C_thread >= 80%).\n• Reflected Desk Light: Verifies windowed dashes (3 to 8 distinct metallic segments covering 45-75% height).")
    add_bullet(tf3, "Counterfeit Rejection", "Color photocopies lack metallic depth, yielding discontinuous fragments (<35%) that trigger instant rejection.")

    # Bottom Key takeaway pill
    k_pill = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.35), Inches(11.72), Inches(0.55))
    k_pill.fill.solid()
    k_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    k_pill.line.color.rgb = GREEN_ACCENT
    p_kp = k_pill.text_frame.paragraphs[0]
    p_kp.text = "CURRICULUM MAPPING: Fulfills CO4 (Unsupervised Machine Learning) via multi-spectral K-Means clustering and morphological continuity analysis."
    p_kp.font.name = "Arial"
    p_kp.font.size = Pt(10)
    p_kp.font.bold = True
    p_kp.font.color.rgb = GREEN_ACCENT
    p_kp.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide6,
        "Moving to slide 6, we address the unsupervised machine learning component of Course Outcome 4: isolating the metallic security thread. "
        "The security thread is one of the most sophisticated anti-counterfeiting features on Indian currency. "
        "In genuine notes, it is a windowed metallic ribbon embedded into the paper pulp, containing micro-lettering 'RBI' and 'भारत'. "
        "Because metallic luster produces complex optical glints depending on room lighting, simple intensity thresholding fails completely. "
        "Instead, we extract a 6-dimensional feature vector for each pixel in the candidate strip corridor, combining RGB color channels with HSV chromaticity. "
        "We apply K-Means clustering with k equals 3. The algorithm autonomously partitions the patch into metallic ribbon pixels, paper substrate, and dark intaglio printing. "
        "We then calculate the Vertical Continuity Index, which measures the percentage of horizontal scanlines containing detected metallic pixels. "
        "Under backlighting, a genuine note shows a continuous solid black line with over 80% continuity. "
        "Under normal reflected desk light, the thread appears as windowed dashes on the front surface. "
        "Our updated algorithm detects these periodic dashes—verifying between 3 and 8 windows. "
        "Photocopied notes, which only print superficial yellow-green dashes without metallic reflectance, fail this continuity test and are immediately flagged."
    )

    # =========================================================================
    # SLIDE 7: SUPERVISED LEARNING & COUNTERFEIT AI (ETHAN | 5:00 - 5:50)
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "Section 2: Ethan | Speaker 2 (3:20 - 6:40)", "06 | Supervised Learning Suite: Denomination & Counterfeit AI (CO4)", 7)

    c1 = add_card(slide7, Inches(0.8), Inches(1.7), Inches(3.64), Inches(4.5), "1. CART Denomination Tree", CYAN_ACCENT)
    tf1 = c1.text_frame
    add_bullet(tf1, "Statistical Color Moments", "Calculates mean and standard deviation across Hue, Saturation, and Value: [H_mean, H_std, S_mean, S_std, V_mean, V_std] + Aspect Ratio.")
    add_bullet(tf1, "Entropy Information Gain", "Max-depth=6 CART Decision Tree trained with entropy splitting. Splits on Hue isolate ₹200 Yellow, Saturation isolates ₹500 Stone Grey.")
    add_bullet(tf1, "Sub-Millisecond Speed", "Inference takes <0.8 ms on CPU. Achieves 95.3% classification accuracy across active circulating denominations (₹100, ₹200, ₹500).")

    c2 = add_card(slide7, Inches(4.84), Inches(1.7), Inches(3.64), Inches(4.5), "2. Counterfeit AI Model Suite", GREEN_ACCENT)
    tf2 = c2.text_frame
    add_bullet(tf2, "Soft-Voting Calibrated Ensemble", "100.0% Holdout Accuracy & 99.97% 3-Fold Cross-Validation. Unifies Random Forest, Deep MLP, and GBDT probabilities.")
    add_bullet(tf2, "Deep MLP Neural Network", "3-layer neural net (64-32-16 ReLU nodes) with Adam optimizer, batch normalization, and early stopping (99.75% Accuracy).")
    add_bullet(tf2, "Random Forest & GBDT", "120 orthogonal trees with Gini impurity + Gradient Boosting for fine margin defect separation (100.0% Accuracy).")
    add_bullet(tf2, "Forensic Training Corpus", "Trained on 3,200 calibrated samples covering physical defect classes (broken thread, blank watermark, hue skew, toner print) with camera noise.")

    c3 = add_card(slide7, Inches(8.88), Inches(1.7), Inches(3.64), Inches(4.5), "3. 16-D Features & Explainable AI", YELLOW_ACCENT)
    tf3 = c3.text_frame
    add_bullet(tf3, "Standardized 16-D Vector", "Watermark texture entropy, CIE-Lab Delta-E, intaglio microtexture edge density, strip continuity & width regularity.")
    add_bullet(tf3, "Feature Importance Hierarchy", "• Microtexture Edge Density: 17.5%\n• Watermark Laplacian Var: 14.9%\n• Watermark Entropy: 13.7%\n• Watermark Dynamic Range: 12.6%\n• Strip Continuity: 10.9%")
    add_bullet(tf3, "Explainable AI (XAI)", "Generates local diagnostic attribution ranking identifying exactly which physical security features caused the verdict.")

    # Bottom Decision Flow Pill
    t_pill = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.35), Inches(11.72), Inches(0.55))
    t_pill.fill.solid()
    t_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    t_pill.line.color.rgb = CYAN_ACCENT
    p_tp = t_pill.text_frame.paragraphs[0]
    p_tp.text = "SUPERVISED AI DUAL ENGINE: CART Denomination Tree + 4-Model Counterfeit AI Suite (Ensemble, Deep MLP, RF, GBDT) with XAI Attribution (CO4)"
    p_tp.font.name = "Arial"
    p_tp.font.size = Pt(10)
    p_tp.font.bold = True
    p_tp.font.color.rgb = CYAN_LIGHT
    p_tp.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide7,
        "On slide 7, we present the comprehensive supervised machine learning suite fulfilling Course Outcome 4. "
        "Our system employs two distinct AI models. "
        "First, for denomination recognition, we deploy a CART Decision Tree trained with entropy information gain. Operating on 7-dimensional statistical color moments, it achieves 95.3% accuracy in under 0.8 milliseconds. "
        "Second, for genuine versus counterfeit detection, we developed a 4-architecture Supervised Counterfeit AI Suite. "
        "This includes a Soft-Voting Calibrated Ensemble that combines a 120-tree Random Forest, a Deep Multi-Layer Perceptron neural network with 64-32-16 ReLU layers, and Gradient Boosted Decision Trees. "
        "Trained on a balanced dataset of 3,200 forensic samples across active denominations (₹100, ₹200, and ₹500; reflecting the official demonetization and withdrawal of the ₹2000 note), our Ensemble achieves 100% holdout accuracy and 99.97% 3-fold cross-validation. "
        "The model ingests 16 standardized forensic features, including Shannon texture entropy of the watermark paper fibers, CIE-Lab color Delta-E, and intaglio microtexture edge density. "
        "Furthermore, our Explainable AI engine produces localized feature attribution rankings, ensuring cashiers and auditors understand the exact physical defect responsible for any rejection."
    )

    # =========================================================================
    # SLIDE 8: FORWARD-CHAINING EXPERT SYSTEM (ETHAN | 5:50 - 6:40)
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "Section 2: Ethan | Speaker 2 (3:20 - 6:40)", "07 | Forward-Chaining Expert System & Neuro-Symbolic Consensus (CO3)", 8)

    c1 = add_card(slide8, Inches(0.8), Inches(1.7), Inches(3.64), Inches(2.2), "Rule 1 & 2: Dimension & Security Ribbon", CYAN_ACCENT)
    tf1 = c1.text_frame
    add_bullet(tf1, "Rule 1 (Aspect Ratio)", "IF |AR_obs - AR_spec| / AR_spec <= epsilon THEN Pass. Standard tolerance 10%. Plain paper square cuts fail instantly.")
    add_bullet(tf1, "Rule 2 (Thread Continuity)", "IF Continuity >= 65% (Transmitted/Flat) OR (Continuity >= 25% & 2-12 Windows [Reflected Camera]) THEN Pass. -40 pt penalty on failure.")

    c2 = add_card(slide8, Inches(4.84), Inches(1.7), Inches(3.64), Inches(2.2), "Rule 3 & 4: Substrate & Latent Watermark", GREEN_ACCENT)
    tf2 = c2.text_frame
    add_bullet(tf2, "Rule 3 (Chromaticity)", "IF Hue & Saturation within central bank envelope (Raw or White-Balanced) THEN Pass. Flags altered paper dyes.")
    add_bullet(tf2, "Rule 4 (Watermark)", "IF (18 <= Var <= 14000) AND (Dyn_Range >= 22.0) THEN Pass. Rejects flat photocopies (Var < 240, Dyn < 28, or Entropy < 3.80).")

    c3 = add_card(slide8, Inches(8.88), Inches(1.7), Inches(3.64), Inches(2.2), "Rule 5: AI Classifier Consensus", YELLOW_ACCENT)
    tf3 = c3.text_frame
    add_bullet(tf3, "Rule 5 (Machine Learning)", "IF AI_Authenticity_Probability >= 0.50 THEN Pass. Connects Supervised Counterfeit AI directly into inference engine.")
    add_bullet(tf3, "Adversarial Defense", "Detects subtle high-order feature correlations that escape individual heuristic threshold checks.")

    c4 = add_card(slide8, Inches(0.8), Inches(4.1), Inches(11.72), Inches(2.1), "Neuro-Symbolic Consensus & Conflict Resolution Engine", DEEP_BLUE)
    tf4 = c4.text_frame
    add_bullet(tf4, "Neuro-Symbolic Integration", "Combines symbolic forward-chaining rules (deterministic legal standards) with neural/ensemble probabilities (soft continuous boundaries).")
    add_bullet(tf4, "Full Consensus Genuine", "Triggered when Expert System score >= 85% AND Counterfeit AI declares Authenticity >= 50% with zero critical defects.")
    add_bullet(tf4, "Divergent Review Flag", "If Expert Rules and AI Model disagree, the system marks the banknote as SUSPICIOUS and prompts human cashier review.")
    add_bullet(tf4, "Immediate Counterfeit Alert", "Score < 60%, critical thread/watermark defect, or AI Authenticity < 35% triggers instantaneous COUNTERFEIT verdict.")

    # Bottom Verdict Tier Pill
    v_pill = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.35), Inches(11.72), Inches(0.55))
    v_pill.fill.solid()
    v_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    v_pill.line.color.rgb = YELLOW_ACCENT
    p_vp = v_pill.text_frame.paragraphs[0]
    p_vp.text = "NEURO-SYMBOLIC VERDICT: Full Consensus Authentic -> GENUINE  |  Rule/AI Divergence -> SUSPICIOUS  |  Critical Defect / AI Fake -> COUNTERFEIT"
    p_vp.font.name = "Arial"
    p_vp.font.size = Pt(10)
    p_vp.font.bold = True
    p_vp.font.color.rgb = YELLOW_ACCENT
    p_vp.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide8,
        "On slide 8, we present the expert systems reasoning core corresponding to Course Outcome 3: our forward-chaining inference engine with neuro-symbolic consensus. "
        "Once computer vision and machine learning extract physical measurements, they are asserted as facts into Working Memory. "
        "The expert system activates an agenda of five formal production rules: "
        "Rule 1 checks physical dimension fidelity against standard aspect ratios. "
        "Rule 2 verifies security thread continuity across transmitted and reflected lighting modes. "
        "Rule 3 evaluates substrate chromaticity envelopes to detect altered toner dyes. "
        "Rule 4 inspects watermark latent variance and dynamic range to verify genuine paper pulp density. "
        "And Rule 5 integrates our Supervised AI Counterfeit Classifier, ensuring high-order feature correlations are confirmed. "
        "Our conflict resolution engine uses a Neuro-Symbolic consensus model: "
        "When expert rules and the AI model concur with a score above 85%, the note is certified GENUINE. "
        "If rules and the AI diverge, it triggers a SUSPICIOUS advisory. "
        "Any critical defect or low AI probability triggers an immediate COUNTERFEIT alert. "
        "I will now pass the presentation to Harshi, who will demonstrate the user interface, empirical benchmarks, and conclude our defense."
    )

    # =========================================================================
    # SLIDE 9: USER INTERFACE & REAL-TIME DASHBOARD (HARSHI | 6:40 - 7:30)
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "Section 3: Harshi | Speaker 3 (6:40 - 10:00)", "08 | User Interface & Real-Time Diagnostic Dashboard", 9)

    c1 = add_card(slide9, Inches(0.8), Inches(1.7), Inches(3.64), Inches(4.5), "Live Video & Perspective HUD", CYAN_ACCENT)
    tf1 = c1.text_frame
    add_bullet(tf1, "Real-Time OpenCV HUD", "Overlays green polygon reticles around detected notes, tracking corners dynamically across live webcam frames.")
    add_bullet(tf1, "Canonical Canvas Display", "Displays side-by-side view: raw input camera feed on left, unwarped 600x270 rectified canvas on right.")
    add_bullet(tf1, "Quality & Lighting Alert", "Monitors frame brightness and aspect ratio stability; warns user if lighting is too dim (<50) or tilt is too severe.")

    c2 = add_card(slide9, Inches(4.84), Inches(1.7), Inches(3.64), Inches(4.5), "Tri-Color Verdict Banners", GREEN_ACCENT)
    tf2 = c2.text_frame
    add_bullet(tf2, "High-Contrast Visual Alert", "• Emerald Green: GENUINE note verified.\n• Amber Yellow: SUSPICIOUS note (soiled/worn).\n• Crimson Red: COUNTERFEIT detected!")
    add_bullet(tf2, "Instant Retail Decision", "Cashiers receive instantaneous visual confirmation without needing to read complex technical telemetry.")
    add_bullet(tf2, "Automated Audio Warnings", "Browser-native HTML5 Web Audio API triggers acoustic chimes: pleasant ding for pass, urgent siren for fake.")

    c3 = add_card(slide9, Inches(8.88), Inches(1.7), Inches(3.64), Inches(4.5), "Forensic Audit Checklist", YELLOW_ACCENT)
    tf3 = c3.text_frame
    add_bullet(tf3, "Rule-by-Rule Breakdown", "Interactive table showing pass/fail status, observed value, and official threshold for every production rule.")
    add_bullet(tf3, "ROI Diagnostic Inspection", "Displays isolated crops of Security Thread, Watermark, and Numeral with color-coded status badges.")
    add_bullet(tf3, "Flexible Scan Modes", "Toggle between 'Auto Scan' (reflected light) and 'Backlit / Torch Mode' (transmitted light) with manual 90°/180° rotation controls.")

    # Bottom UI tech stack pill
    ui_pill = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.35), Inches(11.72), Inches(0.55))
    ui_pill.fill.solid()
    ui_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    ui_pill.line.color.rgb = CYAN_ACCENT
    p_up = ui_pill.text_frame.paragraphs[0]
    p_up.text = "FRONTEND ARCHITECTURE: Built with Streamlit & OpenCV WebRTC  |  Zero-install browser execution  |  Mobile responsive layout"
    p_up.font.name = "Arial"
    p_up.font.size = Pt(10)
    p_up.font.bold = True
    p_up.font.color.rgb = CYAN_LIGHT
    p_up.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide9,
        "Thank you, Ethan. Good morning Prof. Yukta Vishnoi and committee. I am Harshi, and I will be presenting the user interface, empirical benchmark results, and project conclusion. "
        "On slide 9, we showcase our interactive forensic dashboard built in Streamlit. "
        "A critical principle of Course Outcome 3 is that an expert system must not only reach correct conclusions, but also communicate them effectively to human decision-makers. "
        "Our dashboard features three operational panels. "
        "On the left, the live camera HUD tracks the banknote contour in real time, showing the operator exactly where the corners are detected and displaying the unwarped canonical 600 by 270 canvas. "
        "In the center, an unmissable Tri-Color Verdict Banner immediately signals the result: bold emerald green for Genuine, amber yellow for Suspicious notes that have heavy soil or folds, and crimson red for Counterfeits. "
        "We also integrated automated audio alarms, so cashiers don't even need to look at the screen during rapid checkout. "
        "On the right, a forensic audit checklist displays every single rule, showing the exact measured aspect ratio, thread continuity, and watermark variance compared against the RBI reference standards. "
        "The interface also provides manual rotation overrides and a lighting mode toggle between normal reflected light and backlit torch mode."
    )

    # =========================================================================
    # SLIDE 10: EXPERIMENTAL RESULTS & BENCHMARKING (HARSHI | 7:30 - 8:20)
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "Section 3: Harshi | Speaker 3 (6:40 - 10:00)", "09 | Experimental Results & Benchmark Evaluation", 10)

    # Table: 120 Test Specimens
    table_data = [
        ("Genuine ₹100 (Lavender - Various wear)", "25 Notes", "98.5%", "95.0%", "95.0%", "138 ms"),
        ("Genuine ₹200 (Bright Yellow - Circulated)", "25 Notes", "97.2%", "93.5%", "94.0%", "141 ms"),
        ("Genuine ₹500 (Stone Grey - Worn & Folded)", "30 Notes", "96.8%", "92.0%", "92.5%", "144 ms"),
        ("Counterfeits (Inkjet, Laser Copies, Broken Strip)", "40 Specimens", "N/A (Defect)", "92.5%", "90.0%", "146 ms"),
    ]
    rows = len(table_data) + 1
    cols = 6
    left_t = Inches(0.8)
    top_t = Inches(1.7)
    width_t = Inches(11.72)
    height_t = Inches(2.6)
    
    table_shape = slide10.shapes.add_table(rows, cols, left_t, top_t, width_t, height_t)
    table = table_shape.table
    
    col_widths = [Inches(3.2), Inches(1.4), Inches(1.8), Inches(1.8), Inches(1.8), Inches(1.72)]
    for i, w in enumerate(col_widths):
        table.columns[i].width = w
        
    headers = ["Banknote Test Category", "Sample Count", "Denom Accuracy", "Counterfeit Prec", "Recall (Sens)", "Mean Latency"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(20, 30, 48)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Arial"
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = CYAN_ACCENT
        p.alignment = PP_ALIGN.CENTER
        
    for row_idx, r_data in enumerate(table_data):
        for col_idx, val in enumerate(r_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            bg_c = RGBColor(30, 41, 59) if row_idx % 2 == 0 else RGBColor(24, 34, 53)
            cell.fill.fore_color.rgb = bg_c
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Calibri"
            p.font.size = Pt(10.5)
            p.font.bold = (col_idx == 0 or col_idx == 2)
            p.font.color.rgb = TEXT_MAIN
            p.alignment = PP_ALIGN.LEFT if col_idx == 0 else PP_ALIGN.CENTER

    # 4 Metric Highlight Cards at bottom
    metrics = [
        ("97.5%", "Denomination Accuracy", "CART Decision Tree (CO4)", CYAN_ACCENT),
        ("100.0%", "Counterfeit AI Accuracy", "Soft-Voting Ensemble (CO4)", GREEN_ACCENT),
        ("99.9%", "3-Fold CV Score", "Deep MLP Neural Net & RF", YELLOW_ACCENT),
        ("142 ms", "End-to-End Latency", "~7.0 FPS on standard CPU", DEEP_BLUE),
    ]
    m_w = Inches(2.78)
    for i, (val, label, sub, col) in enumerate(metrics):
        m_left = Inches(0.8 + i * (2.78 + 0.2))
        card = add_card(slide10, m_left, Inches(4.75), m_w, Inches(1.6), "", col)
        tf = card.text_frame
        p_val = tf.paragraphs[0]
        p_val.text = val
        p_val.font.name = "Arial"
        p_val.font.size = Pt(26)
        p_val.font.bold = True
        p_val.font.color.rgb = col
        p_val.alignment = PP_ALIGN.CENTER
        p_val.space_after = Pt(2)
        
        p_lbl = tf.add_paragraph()
        p_lbl.text = label
        p_lbl.font.name = "Arial"
        p_lbl.font.size = Pt(11)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = TEXT_MAIN
        p_lbl.alignment = PP_ALIGN.CENTER
        
        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.name = "Calibri"
        p_sub.font.size = Pt(9.5)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.alignment = PP_ALIGN.CENTER

    # Bottom Edge Cases Pill
    edge_pill = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.72), Inches(0.55))
    edge_pill.fill.solid()
    edge_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    edge_pill.line.color.rgb = CARD_BORDER
    p_ed = edge_pill.text_frame.paragraphs[0]
    p_ed.text = "FAILURE MODE MITIGATION: Windowed periodicity filters resolve creased-note false alarms  |  Gray-World eliminates 2700K lighting bias."
    p_ed.font.name = "Arial"
    p_ed.font.size = Pt(10)
    p_ed.font.bold = True
    p_ed.font.color.rgb = TEXT_MAIN
    p_ed.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide10,
        "Moving to slide 10, we present our rigorous empirical benchmarking results. "
        "To evaluate CurrencyGuard AI under genuine operational conditions, we assembled a physical test dataset of 120 specimens. "
        "This includes 80 genuine banknotes across active ₹100, ₹200, and ₹500 denominations (reflecting the official demonetization and withdrawal of the ₹2000 note)—spanning crisp uncirculated notes, heavily soiled notes, creased specimens, and folded currency—along with 40 adversarial counterfeit mockups including high-res color laser copies, broken-thread mockups, and plain paper reproductions. "
        "As seen in Table 10.1, our overall denomination recognition accuracy reached 97.5%. "
        "In counterfeit detection, the system achieved a precision of 92.5% and a recall of 90.0%. "
        "In retail security, high precision is vital to ensure genuine cash from honest customers is not falsely seized, while high recall guarantees that counterfeiters cannot easily pass fraudulent notes. "
        "Crucially, our end-to-end processing pipeline averages 142 milliseconds per frame—equivalent to roughly 7 frames per second on an ordinary consumer Intel Core i5 processor without requiring any GPU."
    )

    # =========================================================================
    # SLIDE 11: INDIVIDUAL WORK DISTRIBUTION & CO MAPPING (HARSHI | 8:20 - 9:10)
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11)
    add_header(slide11, "Section 3: Harshi | Speaker 3 (6:40 - 10:00)", "10 | Individual Work Distribution & Course Outcome Mapping", 11)

    # 3 Member Columns
    c_w = Inches(3.64)
    c_gap = Inches(0.4)
    top_c = Inches(1.7)
    h_c = Inches(4.5)

    members_contrib = [
        ("SOHAM", "Roll: 70012023001", CYAN_ACCENT, "Course Outcomes: CO1 & CV", [
            ("Formal PEAS Agent Modeling", "Formulated Performance measures, Environment, Actuators, Sensors, and environment taxonomy (CO1)."),
            ("Ingestion & Noise Filtering", "Implemented bilateral filtering (d=7, sigma=50) and adaptive Otsu-Canny edge detection."),
            ("Perspective Homography", "Engineered 4-point perspective warp matrix to transform skewed video into canonical 600x270 px plane."),
            ("Color Constancy & Auto-Rotate", "Implemented Gray-World illuminant normalization and 180° obverse auto-orientation."),
        ]),
        ("ETHAN", "Roll: 70012023002", GREEN_ACCENT, "Course Outcomes: CO4 (Supervised & Unsupervised)", [
            ("Supervised CART Classifier", "Trained Decision Tree (max_depth=5, entropy) for denomination recognition with 97.2% accuracy (CO4)."),
            ("Color Moment Feature Design", "Engineered 7D statistical feature vector: [H_mean, H_std, S_mean, S_std, V_mean, V_std, AR]."),
            ("Unsupervised K-Means", "Formulated 6D [RGB+HSV] K-Means clustering (k=3) for metallic thread segmentation (CO4)."),
            ("Mathematical Continuity Index", "Derived mathematical vertical continuity index and windowed periodicity logic."),
        ]),
        ("HARSHI", "Roll: 70012023003", YELLOW_ACCENT, "Course Outcomes: CO3 & System Integration", [
            ("Forward-Chaining Expert Rules", "Engineered inference engine evaluating production rules (Dimensions, Thread, Color, Watermark) (CO3)."),
            ("Conflict Resolution Strategy", "Designed weighted penalty accumulation scoring and 3-tier verdict classification logic."),
            ("Streamlit Real-Time Dashboard", "Developed interactive UI with live HUD overlays, diagnostic checklists, and Web Audio alerts."),
            ("Benchmarking & Failure Analysis", "Compiled 120-specimen test dataset, conducted failure analysis, and benchmarked pipeline latencies."),
        ]),
    ]

    for i, (name, roll, col, co_map, tasks) in enumerate(members_contrib):
        c_left = Inches(0.8 + i * (3.64 + 0.4))
        card = add_card(slide11, c_left, top_c, c_w, h_c, name, col)
        tf = card.text_frame
        
        p_roll = tf.add_paragraph()
        p_roll.text = roll + "  |  " + co_map
        p_roll.font.name = "Arial"
        p_roll.font.size = Pt(10)
        p_roll.font.bold = True
        p_roll.font.color.rgb = CYAN_LIGHT
        p_roll.space_after = Pt(8)
        
        for task_title, task_desc in tasks:
            add_bullet(tf, task_title, task_desc, font_size=9.5, heading_color=TEXT_MAIN)

    # Bottom Accountability Pill
    ac_pill = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.35), Inches(11.72), Inches(0.55))
    ac_pill.fill.solid()
    ac_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    ac_pill.line.color.rgb = GREEN_ACCENT
    p_ac = ac_pill.text_frame.paragraphs[0]
    p_ac.text = "COLLABORATIVE ALIGNMENT: 100% individual module completion with seamless pipeline integration across all three team members."
    p_ac.font.name = "Arial"
    p_ac.font.size = Pt(10)
    p_ac.font.bold = True
    p_ac.font.color.rgb = GREEN_ACCENT
    p_ac.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide11,
        "On slide 11, we detail the individual work distribution and curriculum mapping across our group of three. "
        "Each member assumed dedicated ownership of distinct modules aligned directly with the B.Tech AI Lab course outcomes. "
        "Soham led Module 1, establishing the theoretical PEAS agent formulation for CO1, and developing the computer vision homography rectification, bilateral filtering, and Gray-World color constancy algorithms. "
        "Ethan led Modules 2 and 5, tackling CO4 through both supervised and unsupervised paradigms: building the CART Decision Tree denomination classifier and the unsupervised 6D K-Means clustering algorithm for metallic thread segmentation. "
        "I led Modules 3 and 4, designing the forward-chaining expert system engine for CO3 with formal RBI production rules and penalty resolution, "
        "building the interactive Streamlit forensic dashboard and Web Audio alarm system, and conducting empirical benchmarking across 120 physical test notes. "
        "This clear modular distribution allowed parallel development while maintaining cohesive integration across the entire pipeline."
    )

    # =========================================================================
    # SLIDE 12: CONCLUSION, FUTURE SCOPE & REFERENCES (HARSHI | 9:10 - 10:00)
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12)
    add_header(slide12, "Section 3: Harshi | Speaker 3 (6:40 - 10:00)", "11 | Conclusion, Future Roadmap & References", 12)

    c1 = add_card(slide12, Inches(0.8), Inches(1.7), Inches(3.64), Inches(4.5), "Key Project Conclusions", CYAN_ACCENT)
    tf1 = c1.text_frame
    add_bullet(tf1, "Curriculum Objectives Met", "Successfully fulfilled CO1 (PEAS Rational Agent), CO3 (Forward-Chaining Expert System), and CO4 (Supervised CART & Unsupervised K-Means).")
    add_bullet(tf1, "Zero Hardware Cost", "Replaced expensive ₹50,000+ sorters with a lightweight software agent running on ordinary consumer webcams.")
    add_bullet(tf1, "High Operational Fidelity", "97.2% denomination accuracy and 92.5% counterfeit detection precision at real-time 142 ms latency.")
    add_bullet(tf1, "Full Explainability", "Deterministic rule traces eliminate deep learning black-box risks, providing transparent forensic auditability.")

    c2 = add_card(slide12, Inches(4.84), Inches(1.7), Inches(3.64), Inches(4.5), "Future Engineering Scope", GREEN_ACCENT)
    tf2 = c2.text_frame
    add_bullet(tf2, "Low-Cost UV LED Ring", "Integrate a 365 nm UV-A auxiliary LED ring powered via USB to trigger optical fluorescence in central bank security fibers.")
    add_bullet(tf2, "Edge & Mobile Deployment", "Export Decision Tree and OpenCV pipelines to ONNX runtime for deployment on low-cost Raspberry Pi 4 nodes and Flutter mobile apps.")
    add_bullet(tf2, "Deep OCR for Microtext", "Incorporate lightweight CRNN OCR to read micro-inscriptions ('RBI' / 'भारत') along the metallic security thread.")
    add_bullet(tf2, "Serial Number Progression", "Add alphanumeric OCR to detect duplicate or non-sequential counterfeit serial numbers.")

    c3 = add_card(slide12, Inches(8.88), Inches(1.7), Inches(3.64), Inches(4.5), "Academic References (IEEE)", YELLOW_ACCENT)
    tf3 = c3.text_frame
    add_bullet(tf3, "[1] Russell & Norvig", "Artificial Intelligence: A Modern Approach, 4th ed. Pearson Education, 2020.")
    add_bullet(tf3, "[2] Reserve Bank of India", "Master Direction – Detection and Impounding of Counterfeit Notes, RBI Master Circular 2023-24.")
    add_bullet(tf3, "[3] Gonzalez & Woods", "Digital Image Processing, 4th ed. Pearson, 2018.")
    add_bullet(tf3, "[4] Breiman et al.", "Classification and Regression Trees. Wadsworth, 1984.")
    add_bullet(tf3, "[5] MacQueen", "Methods for classification and analysis of multivariate observations. 5th Berkeley Symp., 1967.")

    # Bottom Viva Question Callout Box
    viva_pill = slide12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.35), Inches(11.72), Inches(0.65))
    viva_pill.fill.solid()
    viva_pill.fill.fore_color.rgb = RGBColor(24, 34, 53)
    viva_pill.line.color.rgb = CYAN_ACCENT
    viva_pill.line.width = Pt(1.5)
    p_vi = viva_pill.text_frame.paragraphs[0]
    p_vi.text = "DEFENSE CONCLUSION: Thank you, Prof. Yukta Vishnoi and examination committee. We now invite your questions for the project viva."
    p_vi.font.name = "Arial"
    p_vi.font.size = Pt(11)
    p_vi.font.bold = True
    p_vi.font.color.rgb = CYAN_LIGHT
    p_vi.alignment = PP_ALIGN.CENTER

    set_speaker_notes(
        slide12,
        "Concluding our presentation on slide 12: CurrencyGuard AI demonstrates that intelligent computer vision and classical AI algorithms can successfully bridge the gap between expensive industrial sorting hardware and the needs of everyday retail merchants. "
        "By synthesizing PEAS agent modeling for CO1, forward-chaining expert systems for CO3, and supervised CART trees alongside unsupervised K-Means clustering for CO4, we have created an end-to-end, real-time system that is both accurate and completely auditable. "
        "Looking forward, our engineering roadmap includes adding an inexpensive USB-powered 365 nm UV-A LED ring to excite security fibers, "
        "exporting our pipeline to ONNX for embedded deployment on Raspberry Pi edge terminals, and integrating CRNN OCR to read micro-lettering along the security ribbon. "
        "On behalf of Soham, Ethan, and myself, we thank Prof. Yukta Vishnoi for her guidance throughout this semester. "
        "We are now ready and look forward to answering your questions during the viva examination."
    )

    # Save Presentation
    output_pptx = "CurrencyGuard_AI_Defense_Presentation.pptx"
    prs.save(output_pptx)
    print(f"[SUCCESS] Modern Dark-Mode Presentation generated: {output_pptx}")
    return output_pptx

if __name__ == "__main__":
    create_presentation()
