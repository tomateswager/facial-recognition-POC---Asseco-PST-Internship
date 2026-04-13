// Page transition loading bar
(function () {
    var bar = document.createElement('div');
    bar.id = '_nav_bar';
    bar.style.cssText = [
        'position:fixed', 'top:0', 'left:0', 'height:3px', 'width:0',
        'background:linear-gradient(90deg,#6366f1,#8b5cf6,#06b6d4)',
        'z-index:99999', 'transition:width 0.25s ease,opacity 0.4s ease',
        'pointer-events:none', 'border-radius:0 2px 2px 0'
    ].join(';');
    document.documentElement.appendChild(bar);

    function start() {
        bar.style.opacity = '1';
        bar.style.width = '70%';
    }
    function finish() {
        bar.style.width = '100%';
        setTimeout(function () { bar.style.opacity = '0'; bar.style.width = '0'; }, 350);
    }

    // Start bar immediately on any internal link click
    document.addEventListener('click', function (e) {
        var a = e.target.closest('a[href]');
        if (!a) return;
        var href = a.getAttribute('href');
        if (!href || href.startsWith('#') || href.startsWith('javascript') ||
            a.target === '_blank' || href.startsWith('http')) return;
        start();
    });

    // Finish when page is ready
    if (document.readyState === 'complete') {
        finish();
    } else {
        window.addEventListener('load', finish);
    }
})();
