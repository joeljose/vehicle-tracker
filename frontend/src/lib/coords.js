/**
 * Coordinate mapping for object-fit:contain images.
 *
 * All drawing coordinates are stored in original pixel space (e.g. 1920x1080)
 * and converted to/from display coordinates for canvas rendering.
 */

/**
 * Compute the displayed image geometry within an <img> element using object-fit:contain.
 * Returns { scale, offsetX, offsetY, displayWidth, displayHeight }.
 */
export function getContainedImageGeometry(imgElement) {
  const { naturalWidth, naturalHeight, clientWidth, clientHeight } = imgElement;
  if (!naturalWidth || !naturalHeight) {
    return { scale: 1, offsetX: 0, offsetY: 0, containerOffsetX: 0, containerOffsetY: 0, displayWidth: clientWidth, displayHeight: clientHeight };
  }

  const scaleX = clientWidth / naturalWidth;
  const scaleY = clientHeight / naturalHeight;
  const scale = Math.min(scaleX, scaleY);

  const displayWidth = naturalWidth * scale;
  const displayHeight = naturalHeight * scale;
  const offsetX = (clientWidth - displayWidth) / 2;
  const offsetY = (clientHeight - displayHeight) / 2;

  // Image may be centered within its parent container (flexbox centering).
  // The canvas uses absolute inset-0 on the container, so we need the
  // image's offset from the container for correct canvas rendering.
  let containerOffsetX = 0;
  let containerOffsetY = 0;
  const parent = imgElement.parentElement;
  if (parent) {
    const imgRect = imgElement.getBoundingClientRect();
    const parentRect = parent.getBoundingClientRect();
    containerOffsetX = imgRect.left - parentRect.left;
    containerOffsetY = imgRect.top - parentRect.top;
  }

  return { scale, offsetX, offsetY, containerOffsetX, containerOffsetY, displayWidth, displayHeight };
}

/**
 * Convert a mouse event's clientX/clientY to original image pixel coordinates.
 * The canvas fills the container (absolute inset-0) while the image is centered
 * within it, so we must account for the image's offset from the container.
 * Returns { x, y } in image space, or null if click is in letterbox area.
 */
export function clientToImageCoords(clientX, clientY, imgElement) {
  const imgRect = imgElement.getBoundingClientRect();
  const geo = getContainedImageGeometry(imgElement);

  const relX = clientX - imgRect.left - geo.offsetX;
  const relY = clientY - imgRect.top - geo.offsetY;

  if (relX < 0 || relY < 0 || relX > geo.displayWidth || relY > geo.displayHeight) {
    return null; // click in letterbox area
  }

  return {
    x: relX / geo.scale,
    y: relY / geo.scale,
  };
}

/**
 * Convert image pixel coordinates back to display (canvas) coordinates.
 * The canvas fills the container, so we need the image's offset within the
 * container (from flexbox centering) plus any object-fit offset.
 */
export function imageToDisplayCoords(imgX, imgY, geo) {
  return {
    x: imgX * geo.scale + geo.offsetX + geo.containerOffsetX,
    y: imgY * geo.scale + geo.offsetY + geo.containerOffsetY,
  };
}
