
const API_ENDPOINT = "/predict";


document.addEventListener("DOMContentLoaded", () => {
  const form         = document.getElementById("assessForm");
  if (!form) return;

  const steps        = form.querySelectorAll(".form-step");
  const stepDots      = document.querySelectorAll("[data-step-dot]");

  const nameInput     = document.getElementById("nameInput");
  const phoneInput    = document.getElementById("phoneInput");
  const nameError     = document.getElementById("nameError");
  const phoneError    = document.getElementById("phoneError");

  const ageInput        = document.getElementById("ageInput");
  const salaryInput     = document.getElementById("salaryInput");
  const employmentInput = document.getElementById("employmentInput");
  const cibilInput      = document.getElementById("cibilInput");
  const cityTierInput   = document.getElementById("cityTierInput");
  const foirInput       = document.getElementById("foirInput");
  const step2Error      = document.getElementById("step2Error");

  const toStep2Btn    = document.getElementById("toStep2");
  const toStep1Btn    = document.getElementById("toStep1");
  const submitBtn     = document.getElementById("submitBtn");

  const resultBox     = document.getElementById("assessResult");

  // ---- helpers ------------------------------------------------------------

  function goToStep(stepNumber) {
    steps.forEach(fieldset => {
      fieldset.classList.toggle("is-active", fieldset.dataset.step === String(stepNumber));
    });
    stepDots.forEach(dot => {
      dot.classList.toggle("is-active", dot.dataset.stepDot === String(stepNumber));
    });
    resultBox.hidden = true;
    resultBox.innerHTML = "";
  }

  function setFieldError(inputEl, errorEl, message) {
    errorEl.textContent = message || "";
    inputEl.closest(".field")?.classList.toggle("has-error", Boolean(message));
  }

  function isValidName(value) {
    return value.trim().length >= 2;
  }

  // Indian mobile numbers: exactly 10 digits, starting 6-9.
  function isValidPhone(value) {
    return /^[6-9]\d{9}$/.test(value.trim());
  }

  // ---- step 1: validate name + phone, then advance -------------------------

  toStep2Btn.addEventListener("click", () => {
    const nameOk  = isValidName(nameInput.value);
    const phoneOk = isValidPhone(phoneInput.value);

    setFieldError(nameInput, nameError, nameOk ? "" : "Enter your full name.");
    setFieldError(phoneInput, phoneError, phoneOk ? "" : "Enter a valid 10-digit mobile number.");

    if (nameOk && phoneOk) {
      goToStep(2);
    }
  });

  toStep1Btn.addEventListener("click", () => goToStep(1));

  // Let Enter in step 1 fields act like clicking Continue.
  [nameInput, phoneInput].forEach(el => {
    el.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        e.preventDefault();
        toStep2Btn.click();
      }
    });
  });

  // Keep the phone field numeric-only as the person types.
  phoneInput.addEventListener("input", () => {
    phoneInput.value = phoneInput.value.replace(/\D/g, "").slice(0, 10);
  });



  function validateStep2() {
    const age    = Number(ageInput.value);
    const salary = Number(salaryInput.value);
    const cibil  = Number(cibilInput.value);
    const foir   = Number(foirInput.value);

    if (!ageInput.value || age < 18 || age > 100) return "Enter an age between 18 and 100.";
    if (!salaryInput.value || salary <= 0) return "Enter your monthly salary.";
    if (!employmentInput.value) return "Select your employment type.";
    if (!cibilInput.value || cibil < 300 || cibil > 900) return "Enter a CIBIL score between 300 and 900.";
    if (!cityTierInput.value) return "Select your city tier.";
    if (!foirInput.value || foir < 0 || foir > 100) return "Enter a FOIR between 0 and 100.";
    return "";
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.querySelector(".btn-label").textContent = isLoading ? "Checking…" : "See my result";
  }

  function renderResult(state, title, body) {
    resultBox.hidden = false;
    resultBox.innerHTML = `
      <div class="result-panel is-${state}">
        <p class="result-title">${title}</p>
        <p class="result-body">${body}</p>
      </div>
    `;
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const step2Problem = validateStep2();
    step2Error.textContent = step2Problem;
    if (step2Problem) return;

    const payload = {
      name: nameInput.value.trim(),
      phone: phoneInput.value.trim(),
      age: Number(ageInput.value),
      salary: Number(salaryInput.value),
      employment_type: employmentInput.value,
      cibil_score: Number(cibilInput.value),
      city_tier: Number(cityTierInput.value),
      foir_score: Number(foirInput.value),
    };

    setLoading(true);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      // Read the body even on 4xx/5xx — the server sends a useful
      // { "error": "..." } message we want to show, not just throw it away.
      if (!response.ok) {
        let errorMessage = `Server responded with status ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData?.error) errorMessage = errorData.error;
        } catch {
          // body wasn't JSON — fall back to the generic message above
        }
        renderResult("error", "Couldn't check that", errorMessage);
        return;
      }

      // Actual shape returned by server.py's /predict route:

      // { "approved": bool, "probability": number, "score": number, "message": string }
      const data = await response.json();

      if (data.error) {
        renderResult("error", "Couldn't check that", data.error);
        return;
      }

      if (data.approved) {
        renderResult(
          "good",
          "Likely approved",
          data.message || `Based on what you shared, your estimated approval confidence is ${data.score ?? "—"}%.`
        );
      } else {
        renderResult(
          "bad",
          "Unlikely to be approved right now",
          data.message || "A few of the factors you entered are working against you at the moment."
        );
      }
    } catch (err) {
      renderResult(
        "error",
        "Couldn't reach the server",
        "Your details weren't sent. Check your connection and then try again."
      );
      console.error("Check Result submission failed:", err);
    } finally {
      setLoading(false);
    }

  });
});