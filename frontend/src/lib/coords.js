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
    return { scale: 1, offsetX: 0, offsetY: 0, displayWidth: clientWidth, displayHeight: clientHeight };
  }

  const scaleX = clientWidth / naturalWidth;
  const scaleY = clientHeight / naturalHeight;
  const scale = Math.min(scaleX, scaleY);

  const displayWidth = naturalWidth * scale;
  const displayHeight = naturalHeight * scale;
  const offsetX = (clientWidth - displayWidth) / 2;
  const offsetY = (clientHeight - displayHeight) / 2;

  return { scale, offsetX, offsetY, displayWidth, displayHeight };
}

/**
 * Convert a mouse event's clientX/clientY to original image pixel coordinates.
 * Returns { x, y } in image space, or null if click is in letterbox area.
 */
export function clientToImageCoords(clientX, clientY, imgElement) {
  const rect = imgElement.getBoundingClientRect();
  const geo = getContainedImageGeometry(imgElement);

  const relX = clientX - rect.left - geo.offsetX;
  const relY = clientY - rect.top - geo.offsetY;

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
 */
export function imageToDisplayCoords(imgX, imgY, geo) {
  return {
    x: imgX * geo.scale + geo.offsetX,
    y: imgY * geo.scale + geo.offsetY,
  };
}
