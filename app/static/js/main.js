// Navigation Active State
document.addEventListener('DOMContentLoaded', () => {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});

// Navbar Scroll Effect
window.addEventListener('scroll', () => {
    const nav = document.querySelector('nav');
    if (window.scrollY > 20) {
        nav.classList.add('glass-nav');
        nav.classList.add('py-2');
    } else {
        nav.classList.remove('glass-nav');
        nav.classList.remove('py-2');
    }
});

// Common GSAP Animations
gsap.from(".logo-container", {
    duration: 1,
    y: -50,
    opacity: 0,
    ease: "power3.out"
});

gsap.from(".nav-link", {
    duration: 1,
    y: -50,
    opacity: 0,
    stagger: 0.1,
    ease: "power3.out"
});
