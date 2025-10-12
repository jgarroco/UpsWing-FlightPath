import structlog

from sqlalchemy.ext.asyncio import AsyncSession

from app.application.ports import UnitOfWork


logger = structlog.get_logger()


class SQLAlchemyUnitOfWork(UnitOfWork):
    """SQLAlchemy implementation of Unit of Work."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                "UnitOfWork error occurred, rolling back",
                exc_type=exc_type.__name__ if exc_type else None,
                exc_val=str(exc_val) if exc_val else None
            )
            await self.rollback()
        elif not self._committed:
            await self.commit()

    async def commit(self):
        """Commit the current transaction."""
        try:
            await self.session.commit()
            self._committed = True
            logger.debug("UnitOfWork committed successfully")
        except Exception as e:
            logger.error("UnitOfWork commit failed", error=str(e))
            await self.rollback()
            raise

    async def rollback(self):
        """Rollback the current transaction."""
        try:
            await self.session.rollback()
            self._committed = False
            logger.debug("UnitOfWork rolled back successfully")
        except Exception as e:
            logger.error("UnitOfWork rollback failed", error=str(e))
            raise
