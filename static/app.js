// Shared keystroke capture and client-side validation for forms with class "capture-keystrokes"
(function(){
  function attachCapture(formId){
    const form = document.getElementById(formId);
    if(!form) return;
    const typing = form.querySelector('input[name="passphrase"], input[type="text"]#typing_field, input#typing_field');
    const hidden = form.querySelector('input[name="keystrokes"]') || form.querySelector('input#keystrokes');

    let startTime = null;
    let events = [];

    function reset(){
      startTime = null;
      events = [];
      if(hidden) hidden.value = '';
    }

    function onKey(e){
      if (!startTime) startTime = Date.now();
      events.push({ key: e.key, t: Date.now() - startTime, type: e.type });
      if(hidden) hidden.value = JSON.stringify(events);
    }

    if(typing){
      typing.addEventListener('keydown', onKey);
      typing.addEventListener('keyup', onKey);
    }

    form.addEventListener('submit', function(ev){
      // basic validation: ensure keystrokes captured and passphrase length
      const pass = (typing && typing.value) || '';
      if(!events || events.length < 4 || pass.length < 5){
        ev.preventDefault();
        alert('Please type at least 5 characters so the system can capture a representative sample.');
        if(typing) typing.focus();
        return false;
      }
      // keep hidden input populated
      if(hidden) hidden.value = JSON.stringify(events);
      return true;
    });

    // Clear capture when fields change drastically
    if(typing){
      typing.addEventListener('focus', reset);
    }
  }

  document.addEventListener('DOMContentLoaded', function(){
    // attach to known forms
    attachCapture('registerForm');
    attachCapture('authForm');

    // Attach to any other forms using class selector
    document.querySelectorAll('form.capture-keystrokes').forEach((f)=>{
      if(!f.id){
        // ensure unique id for forms without ids
        f.id = 'form_' + Math.random().toString(36).slice(2,9);
      }
      attachCapture(f.id);
    });
  });
})();