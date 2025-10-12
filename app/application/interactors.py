
from decimal import Decimal
from app.domain.entities import (
    PlacementTestSession,
    PlacementTestResponse
)
from app.domain.services.cat_service import CATService
from app.application.ports import (
    PlacementTestSessionRepositoryPort, AssessmentConfigRepositoryPort, 
    ClockService, UnitOfWork
)
from app.application.dto import (
    StartPlacementTestCommand, StartPlacementTestResult,
    SubmitAnswerCommand, SubmitAnswerResult, ProgressDTO
)
from app.application.mappers import map_placement_test_item_to_public_dto
from datetime import  timedelta
from app.domain.exceptions import (
    SessionNotFoundException, 
    InvalidSessionStateError, 
    InvalidResponseError, 
    AssessmentConfigurationNotFoundException, 
    ItemNotFoundException
)

class StartPlacementTestInteractor:
    """Use case interactor for starting a placement test."""
    def __init__(
        self,
        session_repository: PlacementTestSessionRepositoryPort,
        config_repository: AssessmentConfigRepositoryPort,
        cat_service: CATService,  
        clock_service: ClockService,
        uow: UnitOfWork
    ):
        self.session_repository = session_repository
        self.config_repository = config_repository
        self.cat_service = cat_service
        self.clock_service = clock_service
        self.uow = uow

    async def execute(self, command: StartPlacementTestCommand) -> StartPlacementTestResult:
        async with self.uow:
            existing_session = await self.session_repository.get_active_session_by_test_taker_id(
                command.test_taker_id
            )

            if existing_session:
                pending_response = await self.session_repository.get_pending_response(existing_session.id)
                if pending_response:
                    question_entity = await self.session_repository.get_item(pending_response.item_id)
                    if not question_entity:
                        raise ItemNotFoundException(f"Assessment Item not found for pending response {pending_response.item_id}")
                    question_dto = map_placement_test_item_to_public_dto(question_entity)
                    config = await self.config_repository.get_config(existing_session.config_id)
                    if not config:
                        raise AssessmentConfigurationNotFoundException(f"Assessment configuration not found for session {existing_session.id}")
                    progress_dto = ProgressDTO(
                        questions_completed=existing_session.questions_answered,
                        max_questions=config.max_questions or 25,
                        current_ability=float(existing_session.current_ability),
                    )
                    return StartPlacementTestResult(
                        session_id=existing_session.id,
                        first_question=question_dto,
                        progress=progress_dto,
                    )

            config = await self.config_repository.get_config_by_type_and_pathway(
                command.assessment_type, command.learning_pathway
            )
            if not config:
                config = await self.config_repository.get_default_config()
            if not config:
                raise AssessmentConfigurationNotFoundException("No assessment configuration found")

            now = self.clock_service.now()
            expires_at = now + timedelta(minutes=120)  
            session_data = PlacementTestSession(
                id="",  
                test_taker_id=command.test_taker_id,
                config_id=config.id,
                current_ability=Decimal(config.starting_ability),
                standard_error=None,
                questions_answered=0,
                is_complete=False,
                started_at=now,
                completed_at=None,
                expires_at=expires_at
            )
            session = await self.session_repository.create_session(session_data)
            available_items = await self.session_repository.get_items_by_skill_areas(
                config.skill_areas, []
            )
            first_question_entity = await self.cat_service.select_next_question(
                ability=float(config.starting_ability),
                skill_areas=config.skill_areas,
                used_item_ids=[],
                available_items=available_items
            )
            if not first_question_entity:
                raise ItemNotFoundException("No suitable questions available for assessment start")
            first_question_dto = map_placement_test_item_to_public_dto(first_question_entity)
            first_response_data = PlacementTestResponse(
                id="",  
                session_id=session.id,
                item_id=first_question_entity.id,
                response_data={},
                is_correct=None,
                raw_score=None,
                presented_at=self.clock_service.now(),
                submitted_at=None,  
                time_taken=None
            )
            await self.session_repository.create_response(first_response_data)
            progress_dto = ProgressDTO(
                questions_completed=session.questions_answered,
                max_questions=config.max_questions or 25,
                current_ability=float(session.current_ability),
            )
            return StartPlacementTestResult(
                session_id=session.id,
                first_question=first_question_dto,
                progress=progress_dto,
            )

class SubmitAnswerInteractor:
    """Use case interactor for submitting answers in a placement test."""
    def __init__(
        self,
        session_repository: PlacementTestSessionRepositoryPort,
        config_repository: AssessmentConfigRepositoryPort,
        cat_service: CATService,  
        clock_service: ClockService,
        uow: UnitOfWork
    ):
        self.session_repository = session_repository
        self.config_repository = config_repository
        self.cat_service = cat_service
        self.clock_service = clock_service
        self.uow = uow

    async def execute(self, command: SubmitAnswerCommand) -> SubmitAnswerResult:
        async with self.uow:
            session = await self.session_repository.get_session(command.session_id)

            print(f"Session start : {session}")
            now = self.clock_service.now()

            if not session:
                raise SessionNotFoundException(f"Placement test session not found: {command.session_id}")

            if not session.can_accept_answer(now):
                raise InvalidSessionStateError(f"Session {command.session_id} cannot accept answers - invalid state")

            pending_response = await self.session_repository.get_pending_response(command.session_id)

            if not pending_response:
                raise SessionNotFoundException(f"No pending response to answer for session {command.session_id}")

            if not pending_response.has_valid_response(command.response_data):
                raise InvalidResponseError("Invalid response data provided")

            item_entity = await self.session_repository.get_item(pending_response.item_id)

            if not item_entity:
                raise ItemNotFoundException(f"Placement test item not found: {pending_response.item_id}")

            score, is_correct = await self.cat_service.score_response(item_entity, command.response_data)

            pending_response.response_data = command.response_data
            pending_response.time_taken = command.time_taken
            pending_response.is_correct = is_correct
            pending_response.raw_score = Decimal(pending_response.calculate_score())
            pending_response.mark_as_submitted(now)

            await self.session_repository.save_response(pending_response)

            previous_responses = await self.session_repository.get_session_responses(command.session_id)
            previous_items = []

            for response in previous_responses:
                item = await self.session_repository.get_item(response.item_id)
                if item:
                    previous_items.append(item)

            current_item = item_entity  
            current_response_score = score

            new_ability, standard_error = await self.cat_service.process_response(
                responses=previous_responses,
                items=previous_items,
                current_item=current_item,
                current_response_score=current_response_score
            )

            session.increment_questions_answered()
            session.update_ability_estimate(Decimal(str(new_ability)), Decimal(str(standard_error)) if standard_error else None)

            config = await self.config_repository.get_config(session.config_id)
            if not config:
                raise AssessmentConfigurationNotFoundException("No assessment configuration found")

            is_complete = self.cat_service.check_termination_criteria(
                session=session,
                config=config,
            )

            if is_complete:
                session.mark_complete(now)

            await self.session_repository.save_session(session)

            if not session.is_complete:
                submitted_responses = await self.session_repository.get_session_responses(command.session_id)
                answered_question_ids = [r.item_id for r in submitted_responses] + [pending_response.item_id]
                available_items = await self.session_repository.get_items_by_skill_areas(
                    config.skill_areas, answered_question_ids
                )
                next_question_entity = await self.cat_service.select_next_question(
                    ability=new_ability,
                    skill_areas=config.skill_areas,
                    used_item_ids=answered_question_ids,
                    available_items=available_items
                )
                if next_question_entity:
                    next_response_data = PlacementTestResponse(
                        id="",
                        session_id=session.id,
                        item_id=next_question_entity.id,
                        response_data={},
                        is_correct=None,
                        raw_score=None,
                        presented_at=self.clock_service.now(),
                        submitted_at=None, 
                        time_taken=None
                    )
                    await self.session_repository.create_response(next_response_data)
                    next_question_dto = map_placement_test_item_to_public_dto(next_question_entity)
                    progress_dto = ProgressDTO(
                        questions_completed=session.questions_answered,
                        max_questions=config.max_questions or 25,
                        current_ability=new_ability,
                        standard_error=Decimal(standard_error) if standard_error else None,
                    )
                    return SubmitAnswerResult(
                        next_question=next_question_dto,
                        progress=progress_dto,
                        is_complete=False,
                        is_correct=is_correct,
                    )
                else:
                    session.mark_complete(now)
                    await self.session_repository.save_session(session)
                    progress_dto = ProgressDTO(
                        questions_completed=session.questions_answered,
                        max_questions=config.max_questions or 25,
                        current_ability=new_ability,
                        standard_error=Decimal(standard_error) if standard_error else None,
                    )
                    return SubmitAnswerResult(
                        next_question=None,
                        progress=progress_dto,
                        is_complete=True,
                        is_correct=is_correct,
                    )
            else:
                progress_dto = ProgressDTO(
                    questions_completed=session.questions_answered,
                    max_questions=config.max_questions or 25,
                    current_ability=new_ability,
                    standard_error=Decimal(standard_error) if standard_error else None,
                )
                return SubmitAnswerResult(
                    next_question=None,
                    progress=progress_dto,
                    is_complete=True,
                    is_correct=is_correct,
                )

