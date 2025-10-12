# UpsWing Assessment System - Business Requirements

## Executive Summary

UpsWing is developing an **Adaptive and Non-Adaptive Assessment System** with **Integrated Learning Recommendations** to enhance the English tutoring platform. This system provides immediate post-assessment course and lesson recommendations for young learners in China, focusing on English as an academic requirement with CEFR-based evaluation.

## Project Overview

### Business Context

- **Target Audience**: Young learners (elementary to 9th grade) in China
- **Primary Purpose**: English assessment for academic requirements (EFL context)
- **Current Need**: Replace existing assessment methods for improved speed, accuracy, and insights
- **Timeline**: 2-week MVP (adaptive assessment priority), then stable development plan

### System Architecture

- **Pattern**: Backend for Frontend (BFF) - Portal Backend aggregates assessment service calls
- **Assessment Service**: Backend APIs providing stateful assessment sessions and immediate recommendations
- **Integration**: Portal manages authentication and user experience, assessment service handles psychometric logic
- **Data Flow**: Student → Portal Frontend → Portal Backend → Assessment API (on-demand requests)

### Team Structure

- **Assessment Team**: Backend assessment APIs, recommendation logic, database design, admin content management
- **Portal Team**: Frontend integration, user experience, authentication management
- **Academic Team**: Content creation and calibration via admin APIs

## Assessment Types

### 1. Adaptive Assessment System

#### 1.1 Core Technology

- **Method**: Computerized Adaptive Testing (CAT)
- **Models**:
  - Primary: Item Response Theory (IRT)
  - Future consideration: Multidimensional Item Response Theory (MIRT)
- **Adaptive Algorithm**: Real-time difficulty adjustment based on response patterns

#### 1.2 Functional Requirements

- **Purpose**: Placement and onboarding assessment for new students
- **Dynamic Question Selection**: Algorithm selects next question based on current ability estimate
- **Stateful Sessions**: Server-side session management with database persistence
- **Ability Estimation**: Continuous recalculation of student ability level using IRT
- **Termination Criteria**: Assessment ends when measurement precision is achieved
- **CEFR Mapping**: Ability scores mapped to CEFR levels (A1-C2)
- **Immediate Recommendations**: Course and lesson suggestions returned with assessment completion

#### 1.3 Question Bank Requirements

- **Item Calibration**: All questions pre-calibrated using IRT parameters
- **Difficulty Range**: Wide spectrum from beginner to advanced levels
- **Content Areas**: Grammar, vocabulary, reading comprehension, listening
- **Quality Assurance**: Psychometric validation of all items

#### 1.4 Session Management and Request-Response Cycle

**Session Lifecycle:**

```
1. Session Creation (POST /assessment/start)
   Portal → Assessment API: {test_taker_id, assessment_type, learning_pathway}
   Assessment API:
     - Create session in assessment_sessions table
     - Initialize MIRT abilities: {"grammar": 0.0, "vocabulary": 0.0, "reading": 0.0}
     - Select initial question using py-irt CAT algorithm
   Response: {session_id, first_question, progress_info}

2. Question Response Cycle (POST /assessment/{session_id}/answer)
   Portal → Assessment API: {answer_data, time_taken}
   Assessment API:
     - Store response in assessment_responses table
     - Update multi-dimensional abilities using py-irt MIRT estimation
     - Select next question based on current ability vector
     - Check termination criteria (standard error threshold for all dimensions)
   Response: {next_question, updated_progress} OR {assessment_complete_signal}

3. Assessment Completion (GET /assessment/{session_id}/complete)
   Portal → Assessment API: Request final results
   Assessment API:
     - Calculate final ability estimates per skill dimension
     - Map abilities to CEFR levels using system configuration
     - Generate skill-specific recommendations based on identified gaps
     - Store results in assessment_results table
   Response: {assessment_results, recommendations, student_profile}
```

**CAT State Management:**

```sql
-- Session table stores single-dimensional ability (for 2PL IRT model)
current_ability DECIMAL(8,4), -- Single theta parameter for 2PL model
standard_error DECIMAL(8,4),  -- Standard error of ability estimate
-- Questions can test multiple skills
skill_areas JSON, -- ["grammar", "vocabulary"]
-- IRT parameters stored in question table
parameters JSON, -- {"difficulty": -1.5, "discrimination": 1.1, "guessing": 0.2}
```

**API Request/Response Examples:**

```json
// Session Start Response
{
  "session_id": "uuid-123",
  "first_question": {
    "question_id": "uuid-456",
    "content": {...},
    "skill_areas": ["grammar", "vocabulary"]
  },
  "progress": {
    "questions_completed": 0,
    "max_questions": 25,
    "current_abilities": {
      "grammar": 0.0,
      "vocabulary": 0.0,
      "reading": 0.0
    }
  }
}

// Answer Response (Continue)
{
  "next_question": {...},
  "progress": {
    "questions_completed": 5,
    "estimated_remaining": 20,
    "current_abilities": {
      "grammar": 0.3,
      "vocabulary": 0.8,
      "reading": 0.5
    }
  }
}

// Final Results Response
{
  "assessment_results": {
    "skill_scores": {
      "grammar": 0.3,
      "vocabulary": 0.8,
      "reading": 0.5
    },
    "cefr_mapping": {
      "grammar": "A2",
      "vocabulary": "B2",
      "reading": "B1"
    },
    "overall_cefr": "B1"
  },
  "recommendations": {
    "student_profile": {
      "strengths": ["vocabulary"],
      "areas_for_improvement": ["grammar"],
      "skill_gaps": {
        "grammar": 0.5,  // gap size: difference from target level
        "vocabulary": 0.1
      }
    },
    "learning_plan": [
      {
        "content_type": "course",
        "title": "Academic Grammar Foundations",
        "target_skill": "grammar",
        "priority": 1,
        "rationale": "Large gap in grammar skills (0.3 vs target 0.8)"
      }
    ]
  }
}
```

**Error Handling & Edge Cases:**

- Session expiry → Automatic cleanup with graceful restart options
- MIRT convergence issues → Fallback to simpler IRT model
- Question bank exhaustion → Intelligent question reuse with different parameters
- Network interruptions → Session state recovery from database

### 2. Non-Adaptive Assessment System

#### 2.1 Purpose and Scope

- **Purpose**: Diagnostic assessment for writing and speaking skills
- **Delivery**: Given at appropriate student level based on adaptive test results or manual assignment
- **Focus**: Academic learning pathway for MVP
- **CEFR Integration**: Results mapped to CEFR levels for consistency

#### 2.2 Speaking Assessments

##### 2.2.1 MVP Scope - Academic Learning Pathway

**Core Evaluation Criteria:**

- **Fluency**: Rate of speech, hesitation patterns, rhythm
- **Pronunciation**: Accuracy of phonemes, stress patterns, intonation
- **Grammar**: Syntactic complexity, accuracy of structures
- **Vocabulary**: Range, precision, academic register
- **Coherence**: Logical flow, discourse markers, topic development
- **Task Completion**: Addressing prompt requirements, content quality

##### 2.1.2 Assessment Format Options

**Option A: General Speaking Assessment**

- Unified evaluation framework incorporating all learning pathways
- Contextual prompts covering academic, social, and professional scenarios
- Single scoring rubric with pathway-specific weighting

**Option B: Academic-Focused Assessment**

- Specialized academic speaking tasks (presentations, discussions, debates)
- Academic vocabulary and register emphasis
- Foundation for future pathway-specific modules

##### 2.2.3 Technical Implementation Options

**Option A: Traditional Audio Processing (MVP Approach)**

- **Audio Recording**: High-quality capture and storage via portal interface
- **Speech-to-Text**: OpenAI Whisper for transcription
- **Feature Extraction**: Custom-built pronunciation and fluency analysis models
- **Limitations**: Requires significant ML development for pronunciation assessment
- **Timeline**: 4-6 weeks for basic implementation

**Option B: Commercial API Integration (Future Consideration)**

- **Services**: SpeechAce, SpeechSuper, or Azure Speech Assessment APIs
- **Benefits**: Professional-grade pronunciation evaluation, immediate implementation
- **Considerations**: Cost implications, dependency on external services
- **Timeline**: 1-2 weeks for integration

**Recommended MVP Strategy**: Start with Whisper transcription + basic fluency metrics, plan commercial API integration for production

#### 2.3 Writing Assessments

##### 2.3.1 Assessment Design Strategy

**Skill-Gap and CEFR-Targeted Essays:**

- **Dynamic Prompt Generation**: Essays tailored to identified skill weaknesses (argumentation, vocabulary, organization)
- **CEFR-Anchored Tasks**: Assessment complexity matched to student's current proficiency level
- **Academic Focus**: Argumentative essays, expository writing, research-based tasks
- **Hybrid Anchoring System**: Combines targeted prompts with exemplar essays for consistent scoring

**Example Assessment Structure:**

```
Student Profile: B1 level, weak in academic argumentation
Generated Assessment:
- Prompt: "Should universities require mandatory community service?"
- Skill Focus: Thesis development, supporting arguments, counterargument addressing
- Exemplar Package: High/Mid/Low B1 argumentation samples
- Weighted Rubric: 40% argumentation, 30% language use, 20% organization, 10% mechanics
```

##### 2.3.2 Technical Implementation Approaches

**Option A: LLM-Based Assessment (Recommended)**

- **Models**: Fine-tuned Llama 3.1 8B or Mistral 7B for academic writing assessment
- **Vector Embeddings**: RAG system with exemplar essays for contextual evaluation
- **Reliability Measures**: Multi-run consistency validation, bias detection algorithms
- **Human Validation**: 10% random sampling for expert review and calibration
- **Benefits**: Sophisticated content evaluation, contextual feedback, scalable

**Option B: Traditional NLP Pipeline (Alternative)**

- **Tools**: spaCy + NLTK + LanguageTool integration
- **Capabilities**: Grammar checking, linguistic analysis, basic structure detection
- **Limitations**: Requires custom academic assessment logic development
- **Benefits**: Fast processing, predictable outputs, no AI model dependencies

**Option C: Hybrid Approach (Production Consideration)**

- **Fast Analysis**: Traditional NLP for grammar, mechanics, basic metrics
- **Deep Assessment**: LLM for content evaluation, argumentation analysis, academic register
- **Benefits**: Combines speed and sophistication, redundant validation
- **Complexity**: Multiple system integration, higher maintenance overhead

##### 2.3.3 Quality Assurance Framework

**Multi-Layer Validation System:**

- **Consistency Validation**: Multiple assessment runs with low temperature settings (≤0.1)
- **Bias Detection**: Statistical analysis across demographic groups, positional bias mitigation
- **Calibration Monitoring**: Target QWK ≥ 0.7 agreement with human expert raters
- **Continuous Learning**: Weekly expert review sampling, model recalibration triggers
- **Fairness Metrics**: No significant score differences across cultural backgrounds for equivalent quality essays

## Learning Pathway Framework

### 3.1 MVP Learning Pathway: Academic

#### 3.1.1 Target Audience

- University-bound students
- Academic English learners
- IELTS/TOEFL preparation students

#### 3.1.2 Core Competencies

- **Academic Vocabulary**: Discipline-specific terminology
- **Critical Thinking**: Analysis, synthesis, evaluation
- **Research Skills**: Source evaluation, citation practices
- **Formal Communication**: Academic presentations, written reports
- **Test-Taking Strategies**: Standardized exam preparation

#### 3.1.3 Assessment Integration

- Adaptive tests focus on academic language structures
- Speaking assessments emphasize formal communication
- Writing assessments prioritize academic genres

### 3.2 Future Learning Pathways

#### 3.2.1 Life & Social Pathway

- Conversational fluency
- Cultural communication
- Social interaction skills
- Everyday vocabulary

#### 3.2.2 Career Pathway

- Professional communication
- Industry-specific vocabulary
- Business writing skills
- Interview preparation

#### 3.2.3 Exam Preparation Pathway

- Test-specific strategies
- Format familiarization
- Time management skills
- Score optimization techniques

## Integrated Learning Recommendations

### 4.1 Recommendation System Architecture

#### 4.1.1 Integration Approach

- **Unified Response Format**: Both adaptive and diagnostic assessments return identical recommendation structure
- **Comprehensive Insights**: MIRT-enabled adaptive assessments provide detailed skill breakdowns comparable to diagnostic tests
- **Immediate Delivery**: Recommendations included in assessment completion API response
- **Content Mapping**: Course and lesson recommendations based on CEFR level, skill gaps, and learning pathway

#### 4.1.2 Recommendation Data Sources

- **Assessment Results**: Ability estimates, skill breakdowns, CEFR mapping
- **Content Database**: Courses and lessons mapped to skills, CEFR levels, and learning pathways
- **Gap Analysis**: Identified weaknesses requiring targeted intervention
- **Learning Progression**: Appropriate next steps based on current ability level

#### 4.1.3 Recommendation Output Structure

- **Student Profile**: Strengths, areas for improvement, overall feedback
- **Learning Plan**: Prioritized course/lesson suggestions with rationale
- **Skill-Specific Guidance**: Targeted recommendations for identified gaps
- **Progressive Pathway**: Logical sequence of learning objectives

### 4.2 Progress Tracking

#### 4.2.1 Metrics Dashboard

- **Ability Level Progression**: IRT-based ability estimates over time
- **Skill Development**: Individual competency growth tracking
- **Goal Achievement**: Progress toward learning objectives
- **Engagement Analytics**: Time spent, completion rates, interaction patterns

#### 4.2.2 Adaptive Adjustments

- **Dynamic Goal Setting**: Modify learning objectives based on performance
- **Difficulty Calibration**: Adjust content complexity automatically
- **Pacing Optimization**: Personalized learning speed recommendations
- **Intervention Triggers**: Alert system for students needing additional support

## MVP Implementation Strategy

### 5.1 MVP Implementation Strategy

#### 5.1.1 Phase 1 - Core MVP (2-Week Timeline)

1. **Adaptive Assessment System**: IRT-based CAT for placement/onboarding with stateful session management
2. **CEFR Mapping**: Ability scores mapped to standardized levels with confidence intervals
3. **Integrated Recommendations**: Course/lesson suggestions with student insights using basic algorithms
4. **Admin APIs**: Question bank upload, assessment configuration, and basic analytics
5. **BFF Integration**: Portal Backend authentication and API communication

**Technical Decisions for MVP:**

- **Writing Assessment**: Traditional NLP pipeline (spaCy + LanguageTool) for speed and reliability
- **Speaking Assessment**: Whisper transcription + basic fluency metrics only
- **Recommendation Engine**: Rule-based matching of CEFR levels to course content
- **Database**: PostgreSQL with proper indexing for session management

#### 5.1.2 Phase 2 - Enhanced Assessment (4-6 Weeks Post-MVP)

1. **LLM Writing Assessment**: Fine-tuned Llama 3.1 8B with vector embeddings for contextual evaluation
2. **Advanced Speaking Features**: Custom pronunciation analysis models or commercial API integration
3. **Skill-Gap Targeted Essays**: Dynamic prompt generation with exemplar anchoring system
4. **Quality Assurance System**: Multi-run validation, bias detection, human calibration pipeline
5. **MIRT Implementation**: Multi-dimensional adaptive testing for detailed skill profiling

**Technical Enhancements:**

- **Vector Database**: Chroma or FAISS for exemplar essay similarity matching
- **LLM Infrastructure**: Local deployment with fine-tuning capabilities
- **Reliability Systems**: Automated consistency monitoring and expert review workflows

#### 5.1.3 Production Scale Considerations (3-6 Months)

1. **Multi-pathway Support**: Life & Social, Career pathway modules with pathway-specific rubrics
2. **Advanced Analytics**: Performance forecasting, learning trajectory analysis, intervention triggers
3. **Enterprise Features**: Bulk student imports, institutional dashboards, detailed reporting
4. **Integration Ecosystem**: Third-party content providers, LMS integrations, mobile applications

**Architecture Evolution:**

- **Microservices Migration**: Service mesh implementation for independent scaling
- **Commercial API Integration**: SpeechAce/SpeechSuper for professional-grade speech assessment
- **Advanced ML Pipeline**: Custom models for pronunciation assessment, automated essay scoring optimization
- **Global Deployment**: Multi-region infrastructure with localized content delivery

### 5.2 Technical Architecture Considerations

#### 5.2.1 Authentication & Security

- **BFF Pattern**: Portal Backend acts as trusted proxy for authentication
- **JWT Validation**: Assessment service validates tokens via public key
- **Session Security**: Stateful sessions with secure database storage
- **API Security**: Standard REST API security practices

#### 5.2.2 Data Architecture

- **On-Demand Access**: Portal requests assessment data via API calls
- **Single Source of Truth**: Assessment service owns all assessment-related data
- **Content Management**: Comprehensive admin APIs for question banks, courses, and lessons
- **Recommendation Integration**: Unified response format for all assessment types

#### 5.2.3 Performance & Scalability

- **Database Optimization**: Proper indexing for session management and question retrieval
- **Response Time**: Assessment operations under 2 seconds
- **Future Microservices**: Architecture designed for service mesh migration
- **Horizontal Scaling**: Database and API tier designed for scaling

### 5.3 Success Metrics

#### 5.3.1 Core System Functionality

MVP Delivery Metrics:

- IRT question selection algorithm: Produces next question based on ability estimate
- Session state management: Saves/retrieves assessment state from database
- CEFR mapping: Converts ability score to CEFR level using provided thresholds
- Basic recommendation: Returns course suggestions based on skill gaps
- Admin APIs: Question upload, config changes work without crashing

Success Criteria: All features function without system errors

#### 5.3.2 Performance Benchmarks

Development Performance Targets:

- Assessment API response: <5s (reasonable for development environment)
- Database operations: Queries complete without timeout
- Session management: Create/retrieve session data reliably
- Question bank: Handles academic team's expected volume (500-1000 items)

Success Criteria: System doesn't crash under normal test loads

#### 5.3.3 Integration Success

Portal Integration Metrics:

- Authentication: Successfully validates JWT tokens
- API contracts: Returns expected JSON structure for all endpoints
- Error handling: Provides meaningful error messages when things go wrong
- Data flow: Assessment results can be retrieved

Success Criteria: Portal team can successfully integrate and test

#### 5.3.4 Algorithm Correctness

IRT/CAT Implementation:

- Mathematical functions: IRT calculations produce expected outputs
- Question selection: Algorithm selects items within configured parameters
- Stopping criteria: Assessment terminates when conditions are met
- Score calculation: Final ability estimates are mathematically valid

Success Criteria: Psychometric calculations match expected formulas

#### 5.3.5 Content Management

Admin Functionality:

- Question bank upload: Academic team can add items via API
- Configuration: Assessment parameters can be modified
- Course management: Course/lesson data can be created and linked
- Basic analytics: System provides assessment data and recommendation details

Success Criteria: Academic team can manage content without developer help

## Risk Mitigation

### 6.1 Technical Risks

- **IRT Model Complexity**: Start with simple 1-parameter model, expand gradually
- **AI Scoring Accuracy**: Implement human validation pipeline for quality assurance
- **Scalability Challenges**: Design for horizontal scaling from MVP stage

### 6.2 Educational Risks

- **Assessment Validity**: Conduct pilot studies with educational experts
- **Cultural Bias**: Include diverse content and validation across student populations
- **Over-reliance on Technology**: Maintain human oversight and intervention capabilities
