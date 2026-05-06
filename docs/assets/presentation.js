document.addEventListener("DOMContentLoaded", () => {
  const slides = Array.from(document.querySelectorAll(".slide"));
  let current = 0;

  const updateCurrent = () => {
    const midpoint = window.scrollY + window.innerHeight * 0.45;
    slides.forEach((slide, index) => {
      const top = slide.offsetTop;
      const bottom = top + slide.offsetHeight;
      if (midpoint >= top && midpoint < bottom) {
        current = index;
      }
    });
  };

  const goTo = (index) => {
    const clamped = Math.max(0, Math.min(slides.length - 1, index));
    slides[clamped].scrollIntoView({ behavior: "smooth", block: "start" });
  };

  window.addEventListener("scroll", updateCurrent, { passive: true });
  updateCurrent();

  document.addEventListener("keydown", (event) => {
    if (["ArrowRight", "PageDown", " "].includes(event.key)) {
      event.preventDefault();
      goTo(current + 1);
    }
    if (["ArrowLeft", "PageUp"].includes(event.key)) {
      event.preventDefault();
      goTo(current - 1);
    }
    if (event.key === "Home") {
      event.preventDefault();
      goTo(0);
    }
    if (event.key === "End") {
      event.preventDefault();
      goTo(slides.length - 1);
    }
  });
});
