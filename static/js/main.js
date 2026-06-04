document.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    document.querySelectorAll('.m-bar div').forEach(el => {
      const w = el.style.width; el.style.width = '0';
      requestAnimationFrame(() => setTimeout(() => el.style.width = w, 60));
    });
  }, 100);
});
