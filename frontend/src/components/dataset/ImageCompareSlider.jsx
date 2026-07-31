/** Before/after image comparison slider — drag the divider to reveal the
 *  reference vs the generated image. Pure CSS clip + pointer events. */
import { useCallback, useRef, useState } from 'react';

export default function ImageCompareSlider({ referenceSrc, generatedSrc, alt = 'comparison' }) {
  const [pos, setPos] = useState(50);           // % from the left
  const boxRef = useRef(null);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX) => {
    const box = boxRef.current;
    if (!box) return;
    const rect = box.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.max(0, Math.min(100, pct)));
  }, []);

  const onPointerDown = (e) => {
    dragging.current = true;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    updateFromClientX(e.clientX);
  };
  const onPointerMove = (e) => {
    if (dragging.current) updateFromClientX(e.clientX);
  };
  const stopDrag = () => { dragging.current = false; };

  return (
    <div
      ref={boxRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={stopDrag}
      onPointerLeave={stopDrag}
      className="relative select-none touch-none overflow-hidden rounded-md border border-border cursor-ew-resize"
      style={{ aspectRatio: '1 / 1', maxWidth: 288 }}
      title="Drag left/right to compare"
    >
      {/* Generated image (bottom layer, always visible) */}
      <img src={generatedSrc} alt={alt} draggable={false}
        className="absolute inset-0 h-full w-full object-cover" />
      {/* Reference image (top layer, clipped from the left up to the divider) */}
      <img src={referenceSrc} alt={`${alt} (reference)`} draggable={false}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
      {/* Divider */}
      <div className="absolute top-0 bottom-0 w-0.5 bg-white/90 shadow"
        style={{ left: `${pos}%`, transform: 'translateX(-50%)' }}>
        <span className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 grid h-7 w-7 place-items-center rounded-full bg-white text-slate-800 text-sm font-bold shadow-md">
          ⇔
        </span>
      </div>
      {/* Labels */}
      <span className="absolute left-1.5 top-1.5 rounded bg-black/55 px-1.5 py-0.5 text-[0.625rem] font-semibold text-white">
        Reference
      </span>
      <span className="absolute right-1.5 top-1.5 rounded bg-black/55 px-1.5 py-0.5 text-[0.625rem] font-semibold text-white">
        Generated
      </span>
    </div>
  );
}
