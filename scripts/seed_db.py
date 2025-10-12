import csv
import asyncio
from decimal import Decimal
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.infrastructure.persistence.connection import get_session
from app.infrastructure.persistence.models.assessment import AssessmentItem, AssessmentConfig
from app.infrastructure.persistence.models.learning import LearningPathway


async def seed_database():
    """Seed the database with initial data from items.csv and default configurations."""
    print("Starting database seeding...")
    
    async for session in get_session():
        print("Session established...")
        
        await seed_learning_pathways(session)
        
        await session.flush()
        
        await seed_assessment_config(session)
        
        await session.flush()
        
        await seed_assessment_items(session)
        
        await session.commit()
        print("Database seeding completed successfully!")


async def seed_learning_pathways(session: AsyncSession):
    """Seed default learning pathways."""
    pathways = [
        {"name": "General", "description": "General English Assessment"},
        {"name": "Academic", "description": "Academic English Assessment"},
        {"name": "Career", "description": "Career English Assessment"},
        {"name": "Life & Social", "description": "Life and Social English Assessment"},
    ]
    
    for pathway_data in pathways:
        result = await session.execute(
            select(LearningPathway).where(LearningPathway.name == pathway_data["name"])
        )
        existing = result.scalar_one_or_none()
        
        if not existing:
            pathway = LearningPathway(
                id=str(uuid.uuid4()),
                name=pathway_data["name"],
                description=pathway_data["description"],
                is_active=True
            )
            session.add(pathway)
            print(f"Added learning pathway: {pathway_data['name']}")
        else:
            print(f"Learning pathway already exists: {pathway_data['name']}")


async def seed_assessment_config(session: AsyncSession):
    """Seed default assessment configuration."""
    result = await session.execute(
        select(AssessmentConfig).where(AssessmentConfig.name == "Default")
    )
    existing = result.scalar_one_or_none()
    
    if not existing:
        pathway_result = await session.execute(
            select(LearningPathway).where(LearningPathway.name == "General")
        )
        general_pathway = pathway_result.scalar_one_or_none()
        
        if general_pathway:
            config = AssessmentConfig(
                id=str(uuid.uuid4()),
                learning_pathway_id=general_pathway.id,
                name="Default",
                assessment_type="adaptive",
                starting_ability=Decimal("0.0"),
                max_questions=25,
                min_questions=5,
                stopping_criterion={"standard_error": 0.3},
                skill_areas=["grammar", "vocabulary", "reading"],
                proficiency_range={"min": -2.0, "max": 2.0},
                is_active=True
            )
            session.add(config)
            print("Added default assessment configuration")
        else:
            print("Warning: General pathway not found, cannot create assessment config")
    else:
        print("Default assessment configuration already exists")


async def seed_assessment_items(session: AsyncSession):
    """Seed assessment items from the items.csv file."""
    print("Starting to seed assessment items from CSV...")
    
    with open('items.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            result = await session.execute(
                select(AssessmentItem).where(AssessmentItem.id == row['id'])
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                options_str = row['options'].strip()
                import ast
                options = ast.literal_eval(options_str)
                
                content = {
                    'item': row['question'],
                    'options': options,
                    'instruction': '',
                    'correct_answer': row['correct_answer']
                }
                
                item = AssessmentItem(
                    id=row['id'],
                    content=content,
                    item_type="multiple_choice",
                    skill_area=[row['skill_area']],  
                    target_proficiency_level=row['cefr_level'],
                    parameters={
                        'difficulty': float(row['difficulty']),
                        'discrimination': float(row['discrimination']),
                        'guessing': float(row['guessing'])
                    },
                    is_active=True
                )
                
                session.add(item)
                print(f"Added assessment item: {row['question'][:50]}...")
            else:
                print(f"Assessment item already exists: {row['id']}")
    
    print("Completed seeding assessment items from CSV")


if __name__ == "__main__":
    asyncio.run(seed_database())
