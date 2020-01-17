function window_width(){
   return window.innerWidth
       || document.documentElement.clientWidth
       || document.body.clientWidth
       || 0;
}

function window_height(){
   return window.innerHeight
       || document.documentElement.clientHeight
       || document.body.clientHeight
       || 0;
}
function draw_pix(ctx, x, y, c) {

}
console.log(window_width() + ", " + window_height())      
window.addEventListener('load', function () {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

})
