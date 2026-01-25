"""
Image URL helper utilities.
"""

from typing import Optional


def get_image_url(
    item_data: dict,
    prefer_high_res: bool = False,
    prefer_icon: bool = False,
) -> Optional[str]:
    """Get image URL from item data with fallback chain.

    Args:
        item_data: Item data dictionary containing image URLs
        prefer_high_res: If True, prefer high resolution images
        prefer_icon: If True, prefer icon images for compact display

    Returns:
        Best available image URL or None
    """
    if prefer_icon:
        # Prefer icon for compact display
        return (
            item_data.get("iconLink")
            or item_data.get("iconLinkFallback")
            or item_data.get("imageLink")
            or item_data.get("image512pxLink")
        )
    elif prefer_high_res:
        # Prefer high-res images
        return (
            item_data.get("image512pxLink")
            or item_data.get("image8xLink")
            or item_data.get("imageLink")
            or item_data.get("iconLink")
            or item_data.get("iconLinkFallback")
        )
    else:
        # Prefer standard images
        return (
            item_data.get("imageLink")
            or item_data.get("iconLink")
            or item_data.get("image512pxLink")
            or item_data.get("iconLinkFallback")
        )
