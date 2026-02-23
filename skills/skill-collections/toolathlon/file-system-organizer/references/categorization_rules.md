# File Categorization Rules & Heuristics

Use these patterns to intelligently decide where to move files during organization.

## Category: School
**Destination Base:** `School/`

### Subcategory: Courses_Materials
*   **File Name Patterns:** `course`, `lecture`, `notes`, `schedule`, `syllabus`, `homework`, `assignment`, `exam`, `quiz`, `lab`
*   **Extensions:** `.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`, `.md`, `.txt`, `.jpg`, `.png` (if related to course content)
*   **Examples from Trajectory:**
    *   `Course_Schedule.jpg`
    *   `course_schedule.xls`
    *   `course_model_weight_*.png`
    *   `Machine_Learning_Course_Notes.md`
    *   `Calculus_Final_Review.ppt`

### Subcategory: Graduation_Projects
*   **File Name Patterns:** `graduation`, `thesis`, `dissertation`, `capstone`, `final_project`, `senior_design`
*   **Extensions:** `.doc`, `.docx`, `.pdf`, `.zip`
*   **Examples:** `Graduation_Materials_Notice_202506.doc`

### Subcategory: Applications_Materials
*   **File Name Patterns:** `recommendation`, `reference`, `transcript`, `application`, `personal_statement`
*   **Extensions:** `.pdf`, `.doc`, `.docx`
*   **Examples:** `Recommendation_Letter_1.pdf`, `Recommendation_Letter_2.pdf`

### Subcategory: Language_Exam_Preparation
*   **File Name Patterns:** `listening`, `reading`, `writing`, `speaking`, `toefl`, `ielts`, `exam`, `test`, `practice`
*   **Extensions:** `.mp3`, `.wav`, `.pdf`, `.xlsx`, `.docx`
*   **Examples:** `Listening1-3.mp3`, `exam.xlsx`

## Category: Work
**Destination Base:** `Work/`

### Subcategory: Projects
*   **File Name Patterns:** `project`, `proposal`, `design`, `spec`, `requirements`, `report`
*   **Extensions:** `.pptx`, `.pdf`, `.docx`, `.xlsx`
*   **Examples:** `Product_Design_Proposal.pptx`

### Subcategory: Software
*   **File Name Patterns:** `setup`, `install`, `.dmg`, `.exe`, `.app`, `software`, `tool`, `utility`
*   **Extensions:** `.dmg`, `.exe`, `.zip`, `.tar.gz`, `.deb`, `.rpm`
*   **Examples:** `Clash.Verge_2.0.3-alpha_aarch64.dmg`

### Subcategory: Job_Application_Materials
*   **File Name Patterns:** `resume`, `cv`, `cover_letter`, `application`, `portfolio`
*   **Extensions:** `.pdf`, `.docx`, `.xlsx`
*   **Examples:** `cv-gboeing.pdf`, `Internship_application_form.xlsx`

### Subcategory: Offer_Galary
*   **File Name Patterns:** `offer`, `letter`, `contract`, `compensation`
*   **Extensions:** `.pdf`, `.docx`
*   **Examples:** (Empty in trajectory - placeholder for future files)

## Category: Entertainment
**Destination Base:** `Entertainment/`

### Subcategory: Movies
*   **File Name Patterns:** `movie`, `film`, `show`, `episode`, `.s0`, `.e0`, `trailer`
*   **Extensions:** `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`
*   **Examples:** `Movie_The_Wandering_Earth.mp4`, `TV_Show_Friends_S01E01.mkv`

### Subcategory: Music
*   **File Name Patterns:** `music`, `song`, `album`, `track`, artist names
*   **Extensions:** `.mp3`, `.wav`, `.flac`, `.aac`, `.m4a`
*   **Examples:** `Music_Jay_Chou_Best.mp3`

### Subcategory: Pictures
*   **File Name Patterns:** Generic image names, location names, `photo`, `img`, `pic`, `screenshot`
*   **Extensions:** `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.heic`
*   **Sub-subcategory Logic:**
    *   `Landscape`: `mountain`, `lake`, `sunset`, `beach`, `view`, location names.
    *   `Pets`: `cat`, `dog`, `pet`, animal names.
    *   `People`: `portrait`, `family`, `friend`, person names.
*   **Examples:**
    *   `Landscape/`: `mount.png`, `sichuan_lake.png`
    *   `Pets/`: `cat.png`

## Decision Flow
1.  **Check Current Path:** A file inside an existing `Entertainment` folder is very likely media.
2.  **Analyze Filename:** Look for the strongest keyword match across categories.
3.  **Check Extension:** `.mp3` → Music; `.mp4` → Movies; `.pdf` → Likely documents (School/Work).
4.  **Default/Uncertain:** If unclear, place in a logical default (e.g., `School/Courses_Materials` for academic-looking documents, `Work/Projects` for professional documents) or ask the user for clarification.
