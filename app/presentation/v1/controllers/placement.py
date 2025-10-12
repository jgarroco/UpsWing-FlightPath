from fastapi import APIRouter, Depends, HTTPException

from app.application.dto import StartPlacementTestCommand, SubmitAnswerCommand
from app.application.interactors import StartPlacementTestInteractor, SubmitAnswerInteractor
from app.setup.ioc.container import get_start_placement_test_interactor, get_submit_answer_interactor

from app.presentation.v1.schemas.assessment import PlacementStartRequest, PlacementSubmitAnswerRequest, PlacementTestStartResponse, PlacementTestSubmitAnswerResponse

router = APIRouter(prefix="/placement", tags=["V1 Placement API"])

@router.post("/start", response_model=PlacementTestStartResponse)
async def start_test(
    request: PlacementStartRequest,
    interactor: StartPlacementTestInteractor = Depends(get_start_placement_test_interactor),
):
    """
    Endpoint: Start a new test session
    """
    command = StartPlacementTestCommand(
        test_taker_id=request.test_taker_id,
        assessment_type=request.assessment_type,
        learning_pathway=request.learning_pathway,
    )
    
    result = await interactor.execute(command)

    return PlacementTestStartResponse(
        session_id=result.session_id,
        first_question=result.first_question,
        progress=result.progress,
    )


@router.post("/{test_id}/answer", response_model=PlacementTestSubmitAnswerResponse)
async def submit_answer(
    test_id: str,
    request: PlacementSubmitAnswerRequest,
    interactor: SubmitAnswerInteractor = Depends(get_submit_answer_interactor)
):
    """
    Endpoint: Submit answer 
    """
    command = SubmitAnswerCommand(
        session_id=test_id,
        response_data=request.response_data,
        time_taken=request.time_taken,
    )
    
    result = await interactor.execute(command)

    return PlacementTestSubmitAnswerResponse(
        next_question=result.next_question,
        progress=result.progress,
        assessment_complete=result.is_complete,
    )
