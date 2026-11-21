
const form = document.getElementById("predictionForm");

form.addEventListener("submit", function () {

    const button = document.getElementById("predictBtn");

    button.innerText = "Analyzing...";
    button.disabled = true;

});
