const BRAND_GRADIENT_STOPS = [
  { offset: '0%', color: '#60a7ce' },
  { offset: '100%', color: '#51b890' },
];

function applyBrandIconGradient(root = document) {
  root.querySelectorAll('.action-card__icon-slot svg').forEach((svg, index) => {
    const gradientId = `brand-gradient-icon-${index}`;
    if (svg.querySelector(`#${gradientId}`)) return;

    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient.setAttribute('id', gradientId);
    gradient.setAttribute('x1', '0%');
    gradient.setAttribute('y1', '0%');
    gradient.setAttribute('x2', '100%');
    gradient.setAttribute('y2', '100%');

    BRAND_GRADIENT_STOPS.forEach(({ offset, color }) => {
      const stop = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
      stop.setAttribute('offset', offset);
      stop.setAttribute('stop-color', color);
      gradient.appendChild(stop);
    });

    defs.appendChild(gradient);
    svg.insertBefore(defs, svg.firstChild);

    const paint = `url(#${gradientId})`;

    if (svg.getAttribute('stroke') !== 'none') {
      svg.setAttribute('stroke', paint);
    }

    const svgFill = svg.getAttribute('fill');
    if (svgFill && svgFill !== 'none' && svgFill !== 'transparent') {
      svg.setAttribute('fill', paint);
    }

    svg.querySelectorAll('path, line, polyline, circle, rect, ellipse').forEach((node) => {
      const stroke = node.getAttribute('stroke');
      const fill = node.getAttribute('fill');

      if (stroke !== 'none') {
        node.setAttribute('stroke', paint);
      }

      if (fill === 'currentColor' || (fill && fill !== 'none' && fill !== 'transparent')) {
        node.setAttribute('fill', paint);
      }
    });
  });
}
