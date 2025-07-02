"""
Domain models for the Voice Assistant application.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class InteractionResult:
    """Result of an interaction attempt"""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class InteractionContext:
    """Context for element interactions"""
    purpose: str
    element_type: str
    action: str
    value: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    selectors: Optional[List[str]] = None
    fallback_selectors: Optional[List[str]] = None
    product_name: Optional[str] = None
    element_id: Optional[str] = None
    element_classes: Optional[List[str]] = None
    timeout: int = 5000  # milliseconds


@dataclass
class PageContext:
    """Context information about the current page"""
    url: str
    title: str
    text: str = ""
    html: str = ""
    input_fields: List[Dict[str, str]] = field(default_factory=list)
    menu_items: List[Dict[str, Any]] = field(default_factory=list)
    buttons: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ActionStep:
    """A step in an interaction action"""
    action: str  # click, type, wait, select, etc.
    selector: str
    value: Optional[str] = None
    description: Optional[str] = None
    fallback_selectors: List[str] = field(default_factory=list)
    timeout: int = 5000  # milliseconds


@dataclass
class VerificationStep:
    """A step to verify an interaction result"""
    type: str  # check_text, check_value, check_state, etc.
    selector: str
    expected: str
    fallback_selectors: List[str] = field(default_factory=list)
    timeout: int = 5000  # milliseconds
