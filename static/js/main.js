document.addEventListener('DOMContentLoaded', () => {
    
    // --- SELECTION DES ELEMENTS -----------------------------------------------
    const zone_depot      = document.getElementById('drop-zone');
    const entree_fichier  = document.getElementById('file-input');
    const apercu_image    = document.getElementById('image-preview');
    const placeholder     = document.getElementById('image-placeholder');
    const overlay_resultat = document.getElementById('result-overlay');
    const texte_prediction = document.getElementById('prediction-text');
    const btn_ok          = document.getElementById('btn-ok');
    const btn_nok         = document.getElementById('btn-nok');
    const msg_correction  = document.getElementById('correction-message');
    const boutons_classes = document.querySelectorAll('.class-btn');
    const btn_stats       = document.getElementById('btn-stats');
    const btn_aide        = document.getElementById('btn-howto');
    const btn_camera      = document.getElementById('btn-camera');
    const flux_camera     = document.getElementById('camera-feed');
    const btn_capture     = document.getElementById('btn-capture');
    const modale_stats    = document.getElementById('modal-stats');
    const conteneur_stats = document.getElementById('stats-container');
    const notification    = document.getElementById('toast');

    let url_image_actuelle = '';
    let mode_correction    = false;
    let flux_video         = null;


    // --- EVENEMENTS ZONE DE DEPOT ---------------------------------------------
    zone_depot.addEventListener('click', () => {
        if (flux_video) {
            capturer_photo();
        } else {
            entree_fichier.click();
        }
    });

    zone_depot.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone_depot.classList.add('drag-over');
    });

    zone_depot.addEventListener('dragleave', () => {
        zone_depot.classList.remove('drag-over');
    });

    zone_depot.addEventListener('drop', (e) => {
        e.preventDefault();
        zone_depot.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            traiter_fichier(e.dataTransfer.files[0]);
        }
    });

    entree_fichier.addEventListener('change', (e) => {
        if (e.target.files.length) {
            traiter_fichier(e.target.files[0]);
        }
    });


    // --- GESTION DES FICHIERS -------------------------------------------------
    function traiter_fichier(fichier) {
        arreter_camera();
        const lecteur = new FileReader();
        lecteur.onload = (e) => {
            apercu_image.src = e.target.result;
            apercu_image.classList.remove('hidden');
            placeholder.classList.add('hidden');
            zone_depot.classList.add('has-image');
            analyser_image(fichier);
        };
        lecteur.readAsDataURL(fichier);
    }


    // --- GESTION CAMERA -------------------------------------------------------
    async function basculer_camera() {
        if (flux_video) {
            arreter_camera();
            placeholder.classList.remove('hidden');
        } else {
            try {
                flux_video = await navigator.mediaDevices.getUserMedia({ 
                    video: { facingMode: 'environment' } 
                });
                flux_camera.srcObject = flux_video;
                flux_camera.classList.remove('hidden');
                placeholder.classList.add('hidden');
                apercu_image.classList.add('hidden');
                btn_capture.classList.remove('hidden');
                overlay_resultat.classList.add('hidden');
            } catch (err) {
                afficher_toast("Acces camera refuse");
                placeholder.classList.remove('hidden');
            }
        }
    }

    function arreter_camera() {
        if (flux_video) {
            flux_video.getTracks().forEach(piste => piste.stop());
            flux_video = null;
            flux_camera.classList.add('hidden');
            btn_capture.classList.add('hidden');
        }
    }

    function capturer_photo() {
        const canevas = document.createElement('canvas');
        canevas.width = flux_camera.videoWidth;
        canevas.height = flux_camera.videoHeight;
        const ctx = canevas.getContext('2d');
        ctx.drawImage(flux_camera, 0, 0);
        
        canevas.toBlob((blob) => {
            const fichier = new File([blob], "capture.jpg", { type: "image/jpeg" });
            apercu_image.src = canevas.toDataURL('image/jpeg');
            apercu_image.classList.remove('hidden');
            flux_camera.classList.add('hidden');
            btn_capture.classList.add('hidden');
            arreter_camera();
            analyser_image(fichier);
        }, 'image/jpeg');
    }


    // --- APPELS API -----------------------------------------------------------
    async function analyser_image(fichier) {
        const formData = new FormData();
        formData.append('image', fichier);
        reset_ui();

        try {
            const reponse = await fetch('/classify', {
                method: 'POST',
                body: formData
            });
            const data = await reponse.json();
            
            if (data.error) throw new Error(data.error);

            url_image_actuelle = data.image_url;
            texte_prediction.innerHTML = `Analyse : <strong>${data.prediction}</strong> (${(data.confidence * 100).toFixed(0)}%)`;
            overlay_resultat.classList.remove('hidden');
            maj_boutons_classes(data.prediction);
            
        } catch (erreur) {
            afficher_toast('Erreur : ' + erreur.message);
        }
    }

    async function envoyer_correction(vrai_label) {
        retour_initial();
        afficher_toast('Merci !');

        try {
            await fetch('/correct', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_url: url_image_actuelle,
                    true_label: vrai_label
                })
            });
        } catch (erreur) {
            afficher_toast('Erreur sauvegarde');
        }
    }

    async function recup_stats() {
        try {
            const reponse = await fetch('/stats');
            const data = await reponse.json();
            
            conteneur_stats.innerHTML = '';
            for (const [label, count] of Object.entries(data)) {
                conteneur_stats.innerHTML += `
                    <div class="stat-item">
                        <div class="stat-val">${count}</div>
                        <div class="stat-label">${label}</div>
                    </div>
                `;
            }
            modale_stats.classList.remove('hidden');
        } catch (erreur) {
            afficher_toast('Erreur stats');
        }
    }


    // --- INTERACTIONS UI ------------------------------------------------------
    overlay_resultat.addEventListener('click', (e) => e.stopPropagation());

    btn_ok.addEventListener('click', (e) => {
        e.stopPropagation();
        afficher_toast('Merci !');
        retour_initial();
    });

    btn_nok.addEventListener('click', (e) => {
        e.stopPropagation();
        mode_correction = true;
        msg_correction.classList.remove('hidden');
        btn_ok.classList.add('hidden');
        btn_nok.classList.add('hidden');
        boutons_classes.forEach(btn => btn.classList.add('highlight'));
    });

    boutons_classes.forEach(btn => {
        btn.addEventListener('click', () => {
            if (mode_correction) {
                envoyer_correction(btn.dataset.label);
            }
        });
    });

    btn_stats.addEventListener('click', recup_stats);
    
    btn_aide.addEventListener('click', () => {
        document.getElementById('modal-howto').classList.remove('hidden');
    });

    btn_camera.addEventListener('click', basculer_camera);
    
    btn_capture.addEventListener('click', (e) => {
        e.stopPropagation();
        capturer_photo();
    });

    // Fonction globale pour les modales (appelee par le HTML)
    window.fermerModale = (id) => {
        document.getElementById(id).classList.add('hidden');
    };

    document.querySelectorAll('.modal').forEach(m => {
        m.addEventListener('click', (e) => {
            if (e.target === m) {
                m.classList.add('hidden');
            }
        });
    });


    // --- UTILITAIRES ----------------------------------------------------------
    function reset_ui() {
        overlay_resultat.classList.add('hidden');
        msg_correction.classList.add('hidden');
        btn_ok.classList.remove('hidden');
        btn_nok.classList.remove('hidden');
        mode_correction = false;
        boutons_classes.forEach(btn => btn.classList.remove('active', 'highlight'));
    }

    function retour_initial() {
        reset_ui();
        apercu_image.classList.add('hidden');
        zone_depot.classList.remove('has-image');
        apercu_image.src = '';
        url_image_actuelle = '';
        if (!flux_video) {
            basculer_camera();
        }
    }

    function maj_boutons_classes(label_predit) {
        boutons_classes.forEach(btn => {
            if (btn.dataset.label === label_predit) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    function afficher_toast(message) {
        notification.textContent = message;
        notification.classList.remove('hidden');
        if (window.timer_toast) clearTimeout(window.timer_toast);
        window.timer_toast = setTimeout(() => notification.classList.add('hidden'), 3000);
    }

    // Lancement auto de la camera
    basculer_camera();
});
