# Updated Database Schema

Here is the reorganized database schema based on the provided notes.

---

## Core Learning & Assessment Structure

### `learning_pathways`

Offers different learning tracks.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| name | `VARCHAR(50)` | | Name of the pathway (e.g., "Academic", "General") |
| description | `VARCHAR(100)`| | A brief description of the pathway |
| is_active | `BOOLEAN` | | Whether the pathway is currently in use |
| created_at | `TIMESTAMP` | | Timestamp of creation |
| updated_at | `TIMESTAMP` | | Timestamp of last update |

<br>

### `assessment_configs`

Drives the behavior of different assessments.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **learning_pathway_id** | `CHAR(36)` | **FK** | Links to `learning_pathways.id` |
| name | `VARCHAR(100)`| | Name of the configuration (e.g., "Default Adaptive") |
| assessment_type | `VARCHAR(100)`| | Type of assessment (e.g., "adaptive") |
| starting_ability | `DECIMAL` | | Initial ability estimate for the test taker |
| max_questions | `INT` | | Maximum number of questions in the assessment |
| min_questions | `INT` | | Minimum number of questions before stopping |
| stopping_criterion | `JSON` | | Rules for when to end the assessment (e.g., SE threshold) |
| skill_areas | `JSON` | | Skills to be tested (e.g., ["grammar", "reading"]) |
| proficiency_range | `JSON` | | The proficiency levels this config covers (e.g., ["A1", "C2"]) |
| is_active | `BOOLEAN` | | Whether the configuration is currently in use |

---

## Assessment Session & Results

### `assessment_sessions`

Tracks an individual test taker's assessment attempt.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **config_id** | `CHAR(36)` | **FK** | Links to `assessment_configs.id` |
| test_taker_id | `VARCHAR(50)` | | The ID of the user taking the test |
| test_taker_type | `VARCHAR(20)` | | Type of test taker - 'student' or 'teacher' |
| current_ability | `DECIMAL(8,4)`| | The continuously updated ability estimate |
| standard_error | `DECIMAL(8,4)`| | The standard error of the ability estimate |
| questions_answered | `INT` | | Count of questions answered so far |
| is_complete | `BOOLEAN` | | Flag indicating if the session is finished |
| started_at | `TIMESTAMP` | | When the session began |
| completed_at | `TIMESTAMP` | | When the session was completed |
| expires_at | `TIMESTAMP` | | When the session will automatically expire |

<br>

### `placement_results`

Stores the final outcome of a placement assessment.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **result_id** | `CHAR(36)` | **FK** | Links to `result.id` |
| final_ability | `JSON` | | Final ability scores, possibly broken down by skill |
| completion_time | `TIMESTAMP` | | Total time taken to complete the assessment |

---

## Question & Answer Bank

### `assessment_items`

The central repository for all questions (items).

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| content | `JSON` | | The actual content of the question (text, options, etc.) |
| item_type | `VARCHAR(50)` | | Type of item (e.g., "multiple_choice", "speaking_prompt") |
| skill_area | `JSON` | | The skill(s) this item measures |
| target_proficiency_level | `VARCHAR(15)` | | The CEFR level this item is most appropriate for |
| parameters | `JSON` | | Other psychometric parameters (e.g., discrimination, guessing) |
| is_active | `BOOLEAN` | | Whether the item is available for assessments |

<br>

### `assessment_responses`

Records a user's answer to a specific assessment item.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **session_id** | `CHAR(36)` | **FK** | Links to `assessment_sessions.id` |
| **item_id** | `CHAR(36)` | **FK** | Links to `assessment_items.id` |
| response_data | `JSON` | | The user's actual answer |
| is_correct | `BOOLEAN` | | Whether the answer was correct (for MCQs) |
| raw_score | `DECIMAL(5,2)`| | Score for subjective items (0.0 to 1.0) |
| presented_at | `TIMESTAMP` | | When the item was shown to the user |
| submitted_at | `TIMESTAMP` | | When the user submitted their answer |
| time_taken | `INT` | | Time in seconds to answer |

---

## Assessment Results

### `result`

Main results table for all assessment types.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **session_id** | `CHAR(36)` | **FK** | Links to `assessment_sessions.id` |
| proficiency_level | `VARCHAR(10)` | | CEFR proficiency level (e.g., A1, B2) |
| feedback | `TEXT` | | Detailed feedback text |
| validated | `BOOLEAN` | | Whether results have been validated |
| skill_scores | `JSON` | | Skill breakdown scores |
| result_type | `VARCHAR(10)` | | Result type: 'P' (placement), 'S' (speaking), 'W' (writing) |
| standard_error | `DECIMAL(8,4)` | | Standard error of measurement |
| created_at | `TIMESTAMP` | | Timestamp of creation |

<br>

### `placement_results`

Stores the final outcome of a placement assessment.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **result_id** | `CHAR(36)` | **FK** | Links to `result.id` |
| final_ability | `JSON` | | Final ability scores, possibly broken down by skill |
| completion_time | `TIMESTAMP` | | Total time taken to complete the assessment |

<br>

### `speaking_results`

Stores results for speaking assessments.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **result_id** | `CHAR(36)` | **FK** | Links to `result.id` |
| transcript | `TEXT` | | The speech-to-text transcript of the user's response |
| criteria_scores | `JSON` | | Scores broken down by criteria (e.g., fluency, pronunciation) |
| overall_score | `DECIMAL` | | The final aggregated score for the speaking task |

<br>

### `writing_results`

Stores results for writing assessments.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **result_id** | `CHAR(36)` | **FK** | Links to `result.id` |
| essay_text | `TEXT` | | The user's written response |
| criteria_scores | `JSON` | | Scores broken down by criteria (e.g., grammar, organization) |
| overall_score | `DECIMAL` | | The final aggregated score for the writing task |

---

## Content & Recommendations

### `courses`

Catalog of available courses.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **pathway_id** | `CHAR(36)` | **FK** | Links to `learning_pathways.id` |
| title | `VARCHAR(50)` | | Course title |
| description | `VARCHAR(255)`| | Course description |
| course_code | `VARCHAR(50)` | | Internal code for the course |
| target_proficiency_level | `VARCHAR(20)` | | The CEFR level this course is designed for |
| primary_skill | `VARCHAR(50)` | | The main skill this course teaches |
| skill_scores | `JSON` | | Detailed skill mapping |
| estimated_duration_hours | `DECIMAL` | | Estimated time to complete the course |
| difficulty_order | `INT` | | Order of difficulty within a proficiency level |
| prerequisites | `JSON` | | List of course IDs that are prerequisites |
| is_active | `BOOLEAN` | | Whether the course is available |
| created_at | `TIMESTAMP` | | Timestamp of creation |
| updated_at | `TIMESTAMP` | | Timestamp of last update |

<br>

### `lessons`

Individual lessons that make up a course.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **course_id** | `CHAR(36)` | **FK** | Links to `courses.id` |
| title | `VARCHAR(100)`| | Lesson title |
| description | `VARCHAR(255)`| | Lesson description |
| lesson_order | `INT` | | Order of the lesson within the course |
| target_skills | `JSON` | | Specific skills targeted in this lesson |
| learning_objectives | `JSON` | | What the student will learn |
| content_type | `VARCHAR(50)` | | Type of content (e.g., "video", "quiz") |
| relative_difficulty | `DECIMAL` | | Difficulty relative to other lessons in the course |
| estimated_duration_minutes | `DECIMAL` | | Estimated time to complete the lesson |
| is_active | `BOOLEAN` | | Whether the lesson is available |

<br>

### `recommendation_items`

Stores individual recommendations generated after an assessment.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| **id** | `CHAR(36)` | **PK** | Primary Key (UUID) |
| **result_id** | `CHAR(36)` | **FK** | Links to `result.id` (main results table) |
| **content_id** | `CHAR(36)` | **FK** | Links to `courses.id` or `lessons.id` |
| content_type | `VARCHAR(20)` | | 'course' or 'lesson' |
| target_skill | `VARCHAR(50)` | | The specific skill this recommendation addresses |
| skill_gap_size | `DECIMAL` | | The estimated size of the student's skill gap |
| priority_order | `INT` | | The order in which recommendations should be presented |
| created_at | `TIMESTAMP` | | Timestamp of creation |