/**
 * iOS Safari 등 corner-shape 미지원 환경용 스쿼클 clip-path 폴백 (figma-squircle 알고리즘)
 */
(function () {
  const hasNativeSquircle =
    typeof CSS !== 'undefined' && CSS.supports('(corner-shape: squircle)');
  const HOME_SQUIRCLE_SELECTOR =
    '.home-page[data-nav="home"] .action-card, .home-page[data-nav="home"] .conversation-card';
  const GLOBAL_SQUIRCLE_SELECTOR = '.action-card, .conversation-card';

  function toRadians(degrees) {
    return (degrees * Math.PI) / 180;
  }

  function rounded(strings, ...values) {
    return strings.reduce((acc, str, i) => {
      const value = values[i];
      if (typeof value === 'number') {
        return acc + str + value.toFixed(4);
      }
      return acc + str + (value ?? '');
    }, '');
  }

  function getPathParamsForCorner({ cornerRadius, cornerSmoothing, preserveSmoothing, roundingAndSmoothingBudget }) {
    let p = (1 + cornerSmoothing) * cornerRadius;
    if (!preserveSmoothing) {
      const maxCornerSmoothing = roundingAndSmoothingBudget / cornerRadius - 1;
      cornerSmoothing = Math.min(cornerSmoothing, maxCornerSmoothing);
      p = Math.min(p, roundingAndSmoothingBudget);
    }
    const arcMeasure = 90 * (1 - cornerSmoothing);
    const arcSectionLength = Math.sin(toRadians(arcMeasure / 2)) * cornerRadius * Math.sqrt(2);
    const angleAlpha = (90 - arcMeasure) / 2;
    const p3ToP4Distance = cornerRadius * Math.tan(toRadians(angleAlpha / 2));
    const angleBeta = 45 * cornerSmoothing;
    const c = p3ToP4Distance * Math.cos(toRadians(angleBeta));
    const d = c * Math.tan(toRadians(angleBeta));
    let b = (p - arcSectionLength - c - d) / 3;
    let a = 2 * b;
    if (preserveSmoothing && p > roundingAndSmoothingBudget) {
      const p1ToP3MaxDistance = roundingAndSmoothingBudget - d - arcSectionLength - c;
      const minA = p1ToP3MaxDistance / 6;
      const maxB = p1ToP3MaxDistance - minA;
      b = Math.min(b, maxB);
      a = p1ToP3MaxDistance - b;
      p = Math.min(p, roundingAndSmoothingBudget);
    }
    return { a, b, c, d, p, arcSectionLength, cornerRadius };
  }

  function drawTopRightPath({ cornerRadius, a, b, c, d, p, arcSectionLength }) {
    if (cornerRadius) {
      return rounded`c ${a} 0 ${a + b} 0 ${a + b + c} ${d} a ${cornerRadius} ${cornerRadius} 0 0 1 ${arcSectionLength} ${arcSectionLength} c ${d} ${c} ${d} ${b + c} ${d} ${a + b + c}`;
    }
    return rounded`l ${p} 0`;
  }

  function drawBottomRightPath({ cornerRadius, a, b, c, d, p, arcSectionLength }) {
    if (cornerRadius) {
      return rounded`c 0 ${a} 0 ${a + b} ${-d} ${a + b + c} a ${cornerRadius} ${cornerRadius} 0 0 1 -${arcSectionLength} ${arcSectionLength} c ${-c} ${d} ${-(b + c)} ${d} ${-(a + b + c)} ${d}`;
    }
    return rounded`l 0 ${p}`;
  }

  function drawBottomLeftPath({ cornerRadius, a, b, c, d, p, arcSectionLength }) {
    if (cornerRadius) {
      return rounded`c ${-a} 0 ${-(a + b)} 0 ${-(a + b + c)} ${-d} a ${cornerRadius} ${cornerRadius} 0 0 1 -${arcSectionLength} -${arcSectionLength} c ${-d} ${-c} ${-d} ${-(b + c)} ${-d} ${-(a + b + c)}`;
    }
    return rounded`l ${-p} 0`;
  }

  function drawTopLeftPath({ cornerRadius, a, b, c, d, p, arcSectionLength }) {
    if (cornerRadius) {
      return rounded`c 0 ${-a} 0 ${-(a + b)} ${d} ${-(a + b + c)} a ${cornerRadius} ${cornerRadius} 0 0 1 ${arcSectionLength} -${arcSectionLength} c ${c} ${-d} ${b + c} ${-d} ${a + b + c} ${-d}`;
    }
    return rounded`l 0 ${-p}`;
  }

  function getSvgPath({ width, height, cornerRadius, cornerSmoothing, preserveSmoothing = true }) {
    const roundingAndSmoothingBudget = Math.min(width, height) / 2;
    const radius = Math.min(cornerRadius, roundingAndSmoothingBudget);
    const pathParams = getPathParamsForCorner({
      cornerRadius: radius,
      cornerSmoothing,
      preserveSmoothing,
      roundingAndSmoothingBudget,
    });
    return `
      M ${width - pathParams.p} 0
      ${drawTopRightPath(pathParams)}
      L ${width} ${height - pathParams.p}
      ${drawBottomRightPath(pathParams)}
      L ${pathParams.p} ${height}
      ${drawBottomLeftPath(pathParams)}
      L 0 ${pathParams.p}
      ${drawTopLeftPath(pathParams)}
      Z
    `
      .replace(/[\t\s\n]+/g, ' ')
      .trim();
  }

  const CORNER_SMOOTHING = 1;
  let scheduled = false;

  function collectSquircleTargets(scope) {
    const root = scope?.querySelectorAll ? scope : document;
    const nodes = new Set();

    root.querySelectorAll(HOME_SQUIRCLE_SELECTOR).forEach((el) => nodes.add(el));
    if (!hasNativeSquircle) {
      root.querySelectorAll(GLOBAL_SQUIRCLE_SELECTOR).forEach((el) => nodes.add(el));
    }
    if (root.matches?.(HOME_SQUIRCLE_SELECTOR)) {
      nodes.add(root);
    } else if (!hasNativeSquircle && root.matches?.(GLOBAL_SQUIRCLE_SELECTOR)) {
      nodes.add(root);
    }

    return nodes;
  }

  function cornerRadiusFor(el) {
    if (el.classList.contains('action-card')) {
      return 22;
    }

    const cap = Math.min(el.offsetWidth, el.offsetHeight) / 2;
    const parsed = parseFloat(getComputedStyle(el).borderTopLeftRadius);
    if (Number.isFinite(parsed) && parsed > 0) {
      return Math.min(parsed, cap);
    }
    return Math.min(10, cap);
  }

  function applyTo(el) {
    const width = el.offsetWidth;
    const height = el.offsetHeight;
    if (width < 1 || height < 1) return;

    const path = getSvgPath({
      width,
      height,
      cornerRadius: cornerRadiusFor(el),
      cornerSmoothing: CORNER_SMOOTHING,
      preserveSmoothing: true,
    });
    const clip = `path('${path}')`;
    el.style.clipPath = clip;
    el.style.webkitClipPath = clip;
    el.classList.add('squircle-clipped');
  }

  function applySquircleFallback(root) {
    collectSquircleTargets(root).forEach(applyTo);
  }

  function scheduleApply() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      applySquircleFallback();
    });
  }

  window.applySquircleFallback = applySquircleFallback;

  function init() {
    applySquircleFallback();
    if (typeof ResizeObserver !== 'undefined') {
      const resizeObserver = new ResizeObserver(scheduleApply);
      resizeObserver.observe(document.body);
    }
    const mutationObserver = new MutationObserver(scheduleApply);
    mutationObserver.observe(document.body, { childList: true, subtree: true });
    window.addEventListener('resize', scheduleApply, { passive: true });
    window.addEventListener('orientationchange', scheduleApply, { passive: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
