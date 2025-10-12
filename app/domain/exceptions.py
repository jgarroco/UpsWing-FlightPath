"""Problem exception classes that follow RFC 9457 using fastapi-problem."""

from fastapi_problem.error import NotFoundProblem, BadRequestProblem, ServerProblem


class SessionNotFoundException(NotFoundProblem):
    """Problem raised when a placement test session is not found."""
    title = "Session not found"
    type_ = "session-not-found"


class InvalidSessionStateError(BadRequestProblem):
    """Problem raised when a session is in an invalid state for the requested operation."""
    title = "Invalid session state"
    type_ = "invalid-session-state"


class InvalidResponseError(BadRequestProblem):
    """Problem raised when submitted response data is invalid."""
    title = "Invalid response data"
    type_ = "invalid-response"


class AssessmentConfigurationNotFoundException(NotFoundProblem):
    """Problem raised when assessment configuration is not found."""
    title = "Assessment configuration not found"
    type_ = "assessment-configuration-not-found"


class ItemNotFoundException(NotFoundProblem):
    """Problem raised when an assessment item is not found."""
    title = "Assessment item not found"
    type_ = "item-not-found"


class AssessmentTerminatedException(BadRequestProblem):
    """Problem raised when attempting to interact with a terminated assessment session."""
    title = "Assessment terminated"
    type_ = "assessment-terminated"


class InternalServerError(ServerProblem):
    """General server error problem."""
    title = "Internal server error"
    type_ = "internal-server-error"
