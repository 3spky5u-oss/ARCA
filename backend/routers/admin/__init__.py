"""
Admin Router - System Administration API

Provides endpoints for system status, configuration, testing, and monitoring.
Protected by JWT bearer authentication.
"""

import logging

from fastapi import APIRouter
from fastapi.security import HTTPBearer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])
_bearer_scheme = HTTPBearer(auto_error=False)

# Import log_capture first (setup_log_capture runs on import from sub-modules)
from .log_capture import setup_log_capture, BufferedLogHandler, _log_buffer

# Initialize log capture on package load (matches original behavior)
setup_log_capture()

# Re-export ConfigUpdate for any external consumers
from .models import ConfigUpdate

# Import all sub-modules to register their routes on the shared router
from . import auth
from . import users
from . import status
from . import testing
from . import sessions_logs
from . import models_routes
from . import domains
from . import tools
