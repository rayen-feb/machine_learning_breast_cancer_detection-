document.addEventListener('DOMContentLoaded', () => {
  // Sticky navbar shadow on scroll
  const navbar = document.getElementById('navbar');
  if (navbar) {
    const onScroll = () => {
      navbar.classList.toggle('scrolled', window.scrollY > 8);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  // Mobile nav toggle
  const navToggle = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      const isOpen = navLinks.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', String(isOpen));
    });

    navLinks.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        navLinks.classList.remove('open');
        navToggle.setAttribute('aria-expanded', 'false');
      });
    });
  }

  // 3D tilt-on-hover for device mockup, video card, and gallery cards
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!prefersReducedMotion) {
    document.querySelectorAll('.tilt-card').forEach((card) => {
      const strength = 10;
      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5;
        const y = (e.clientY - rect.top) / rect.height - 0.5;
        card.style.setProperty('--rx', `${(x * strength).toFixed(2)}deg`);
        card.style.setProperty('--ry', `${(-y * strength).toFixed(2)}deg`);
      });
      card.addEventListener('mouseleave', () => {
        card.style.setProperty('--rx', '0deg');
        card.style.setProperty('--ry', '0deg');
      });
    });
  }

  // Custom play button over the demo video, with a graceful empty state
  // when static/media/demo.mp4 hasn't been added yet.
  const demoVideo = document.getElementById('demoVideo');
  const videoPlayBtn = document.getElementById('videoPlayBtn');
  const videoEmptyState = document.getElementById('videoEmptyState');
  const videoCard = document.getElementById('videoCard');

  if (demoVideo && videoPlayBtn && videoEmptyState && videoCard) {
    videoPlayBtn.addEventListener('click', () => demoVideo.play());
    demoVideo.addEventListener('play', () => videoPlayBtn.classList.add('is-hidden'));
    demoVideo.addEventListener('pause', () => videoPlayBtn.classList.remove('is-hidden'));
    demoVideo.addEventListener('ended', () => videoPlayBtn.classList.remove('is-hidden'));

    const showEmptyState = () => {
      videoCard.classList.add('video-card--empty');
      videoPlayBtn.classList.add('is-hidden');
    };

    demoVideo.addEventListener('error', showEmptyState);
    // If no source loads at all (readyState stays 0), treat it as missing too.
    setTimeout(() => {
      if (demoVideo.readyState === 0) showEmptyState();
    }, 1200);
  }

  // Scroll-triggered reveal animations
  const revealEls = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && revealEls.length) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry, i) => {
          if (entry.isIntersecting) {
            setTimeout(() => entry.target.classList.add('is-visible'), i * 60);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
    );
    revealEls.forEach((el) => observer.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add('is-visible'));
  }
});
