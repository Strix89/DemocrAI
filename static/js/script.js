const messageForm = document.querySelector(".prompt__form");
const chatHistoryContainer = document.querySelector(".chats");
const suggestionItems = document.querySelectorAll(".suggests__item");

const themeToggleButton = document.getElementById("themeToggler");
const clearChatButton = document.getElementById("deleteButton");

// State variables
let currentUserMessage = null;
let isGeneratingResponse = false;

// API endpoints
const NEW_CHAT_ENDPOINT = `${window.API_BASE_URL}/new_chat`;
const SEND_MESSAGE_ENDPOINT = `${window.API_BASE_URL}/send_message`;

// Initialize localStorage chat ID
const chatIdKey = "chat_id";


// create a new chat message element
const createChatMessageElement = (htmlContent, ...cssClasses) => {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", ...cssClasses);
    messageElement.innerHTML = htmlContent;
    return messageElement;
};

// Show typing effect
const showTypingEffect = (rawText, htmlText, messageElement, incomingMessageElement, skipEffect = false) => {
    const copyIconElement = incomingMessageElement.querySelector(".message__icon");
    copyIconElement.classList.add("hide"); // Initially hide copy button

    if (skipEffect) {
        // Display content directly without typing
        messageElement.innerHTML = htmlText;
        hljs.highlightAll();
        addCopyButtonToCodeBlocks();
        copyIconElement.classList.remove("hide"); // Show copy button
        isGeneratingResponse = false;
        return;
    }

    const wordsArray = rawText.split(' ');
    let wordIndex = 0;

    const typingInterval = setInterval(() => {
        messageElement.innerText += (wordIndex === 0 ? '' : ' ') + wordsArray[wordIndex++];
        if (wordIndex === wordsArray.length) {
            clearInterval(typingInterval);
            isGeneratingResponse = false;
            messageElement.innerHTML = htmlText;
            hljs.highlightAll();
            addCopyButtonToCodeBlocks();
            copyIconElement.classList.remove("hide");
        }
    }, 75);
};

// Add copy button to code blocks
const addCopyButtonToCodeBlocks = () => {
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach((block) => {
        const codeElement = block.querySelector('code');
        let language = [...codeElement.classList].find(cls => cls.startsWith('language-'))?.replace('language-', '') || 'Text';

        const languageLabel = document.createElement('div');
        languageLabel.innerText = language.charAt(0).toUpperCase() + language.slice(1);
        languageLabel.classList.add('code__language-label');
        block.appendChild(languageLabel);

        const copyButton = document.createElement('button');
        copyButton.innerHTML = `<i class='bx bx-copy'></i>`;
        copyButton.classList.add('code__copy-btn');
        block.appendChild(copyButton);

        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(codeElement.innerText).then(() => {
                copyButton.innerHTML = `<i class='bx bx-check'></i>`;
                setTimeout(() => copyButton.innerHTML = `<i class='bx bx-copy'></i>`, 2000);
            }).catch(err => {
                console.error("Copy failed:", err);
                alert("Unable to copy text!");
            });
        });
    });
};

// Show loading animation during API request
const displayLoadingAnimation = () => {
    const loadingHtml = `
        <div class="message__content">
            <img class="message__avatar" src="${window.STATIC_URL_LOGO}" alt="Bot Avatar">
            <p class="message__text"></p>
            <div class="message__loading-indicator">
                <div class="message__loading-bar"></div>
                <div class="message__loading-bar"></div>
                <div class="message__loading-bar"></div>
            </div>
        </div>
        <span class="message__icon hide"><i class='bx bx-copy-alt'></i></span>
    `;
    const loadingMessageElement = createChatMessageElement(loadingHtml, "message--incoming", "message--loading");
    chatHistoryContainer.appendChild(loadingMessageElement);
};

// Copy message to clipboard
const copyMessageToClipboard = (copyButton) => {
    const messageContent = copyButton.parentElement.querySelector(".message__text").innerText;

    navigator.clipboard.writeText(messageContent);
    copyButton.innerHTML = `<i class='bx bx-check'></i>`; // Confirmation icon
    setTimeout(() => copyButton.innerHTML = `<i class='bx bx-copy-alt'></i>`, 1000); // Revert icon after 1 second
};

const handleOutgoingMessage = async () => {
    currentUserMessage = messageForm.querySelector(".prompt__form-input").value.trim() || currentUserMessage;
    if (!currentUserMessage || isGeneratingResponse) return;

    isGeneratingResponse = true;

    const outgoingMessageHtml = `
        <div class="message__content">
            <img class="message__avatar" src="${window.STATIC_URL_AVATAR}" alt="User avatar">
            <p class="message__text"></p>
        </div>
    `;
    const outgoingMessageElement = createChatMessageElement(outgoingMessageHtml, "message--outgoing");
    outgoingMessageElement.querySelector(".message__text").innerText = currentUserMessage;
    chatHistoryContainer.appendChild(outgoingMessageElement);

    messageForm.reset();
    document.body.classList.add("hide-header");

    setTimeout(displayLoadingAnimation, 5);
    try {
        let chatId = localStorage.getItem(chatIdKey);
        if (!chatId) {
            chatId = await createNewChat(currentUserMessage);
        }
        const botResponse = await sendMessageToChat(chatId, currentUserMessage);

        const incomingMessageHtml = `
            <div class="message__content">
                <img class="message__avatar" src="${window.STATIC_URL_LOGO}" alt="Bot Avatar">
                <p class="message__text"></p>
            </div>
            <span onClick="copyMessageToClipboard(this)" class="message__icon hide"><i class="bx bx-copy-alt"></i></span>
        `;
        const incomingMessageElement = createChatMessageElement(incomingMessageHtml, "message--incoming");
        chatHistoryContainer.appendChild(incomingMessageElement);
        const messageElement = incomingMessageElement.querySelector(".message__text");
        showTypingEffect(
            botResponse,      // rawText
            marked.parse(botResponse),      // htmlText
            messageElement,
            incomingMessageElement,
            false             // skipEffect
        );
    } catch (error) {
        console.error("Error handling outgoing message:", error);
    } finally {
        isGeneratingResponse = false;
        const loadingElement = chatHistoryContainer.querySelector(".message--loading");
        if (loadingElement) loadingElement.remove();
    }
};

const createNewChat = async (message) => {
    try {
        const response = await fetch(NEW_CHAT_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });

        if (!response.ok) {
            let errorMessage = "Errore nella creazione della chat sconociuto";
            try {
                const errorData = await response.json();
                errorMessage = errorData.system?.error || errorData.message || errorMessage;
            } catch (jsonError) {
                console.error("Impossibile leggere il JSON dell'errore:", jsonError);
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        localStorage.setItem(chatIdKey, data.system.chat_id);
        return data.system.chat_id;
    } catch (error) {
        console.error("Error creating new chat:", error);
        displayErrorMessage(error);
        throw error;
    }
};

const sendMessageToChat = async (chatId, message) => {
    try {
        const response = await fetch(SEND_MESSAGE_ENDPOINT, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ chat_id: chatId, message })
        });
        if (!response.ok) {
            let errorMessage = "Errore nell'invio del messagio sconosciuto";
            try {
                const errorData = await response.json();
                errorMessage = errorData.system?.error || errorData.message || errorMessage;
            } catch (jsonError) {
                console.error("Impossibile leggere il JSON dell'errore:", jsonError);
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        return data.model.response;
    } catch (error) {
        console.error("Error sending message:", error);
        displayErrorMessage(error);
        throw error;
    } 
};

// Toggle between light and dark themes
themeToggleButton.addEventListener('click', () => {
    const isLightTheme = document.body.classList.toggle("light_mode");
    localStorage.setItem("themeColor", isLightTheme ? "light_mode" : "dark_mode");

    // Update icon based on theme
    const newIconClass = isLightTheme ? "bx bx-moon" : "bx bx-sun";
    themeToggleButton.querySelector("i").className = newIconClass;
});

clearChatButton.addEventListener('click', () => {
    if (confirm("Vuoi cancellare la chat? E aprirne una nuova?")) {
        localStorage.removeItem(chatIdKey);
        messageForm.reset();
        document.body.classList.remove("hide-header");
        chatHistoryContainer.innerHTML = "";
        currentUserMessage = null;
        isGeneratingResponse = false;
    }
});

// Handle click on suggestion items
suggestionItems.forEach(suggestion => {
    suggestion.addEventListener('click', () => {
        currentUserMessage = suggestion.querySelector(".suggests__item-text").innerText;
        handleOutgoingMessage();
    });
});

// Prevent default from submission and handle outgoing message
messageForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleOutgoingMessage();
});

const loadChatHistory = async () => {
    // Gestione del tema
    const isLightTheme = localStorage.getItem("themeColor") === "light_mode";
    document.body.classList.toggle("light_mode", isLightTheme);
    themeToggleButton.innerHTML = isLightTheme
        ? '<i class="bx bx-moon"></i>'
        : '<i class="bx bx-sun"></i>';

    const chatId = localStorage.getItem(chatIdKey);
    if (!chatId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/chats/${chatId}`);
        if (!response.ok) {
            throw new Error("Failed to load chat history.");
        }

        const chatData = await response.json();
        // Svuotiamo la chat prima di ricaricare i messaggi
        chatHistoryContainer.innerHTML = "";

        chatData.messages.forEach((msg) => {
            // --- Messaggio inviato dall'utente ---
            if (msg.sender === "user") {
                const userMessageHtml = `
                    <div class="message__content">
                        <img class="message__avatar"
                             src="${window.STATIC_URL_AVATAR}"
                             alt="User Avatar" />
                        <!-- Mettiamo il testo così com'è (niente Markdown) -->
                        <p class="message__text">${msg.text}</p>
                    </div>
                `;
                const userMessageElement = createChatMessageElement(userMessageHtml, "message--outgoing");
                chatHistoryContainer.appendChild(userMessageElement);
            }
            // --- Messaggio di errore inviato dal "system" ---
            else if (msg.sender === "system") {
                // Niente pulsante di copia, solo il testo in rosso
                const errorMessageHtml = `
                    <div class="message__content">
                        <img class="message__avatar"
                             src="${window.STATIC_URL_LOGO}"
                             alt="Assistant Avatar" />
                        <p class="message__text">${msg.text}</p>
                    </div>
                `;
                const errorMessageElement = createChatMessageElement(errorMessageHtml, "message--incoming");
                // Applichiamo la classe .message--error per lo stile (ad es. sfondo rosso)
                errorMessageElement.classList.add("message--error");
                chatHistoryContainer.appendChild(errorMessageElement);
            }
            // --- Messaggio inviato dal bot (o da un sender diverso) ---
            else {
                const botMessageHtml = `
                    <div class="message__content">
                        <img class="message__avatar"
                             src="${window.STATIC_URL_LOGO}"
                             alt="Bot Avatar" />
                        <p class="message__text"></p>
                    </div>
                    <!-- Pulsante di copia dell'intero messaggio -->
                    <span class="message__icon hide" onclick="copyMessageToClipboard(this)">
                        <i class="bx bx-copy-alt"></i>
                    </span>
                `;
                const botMessageElement = createChatMessageElement(botMessageHtml, "message--incoming");
                chatHistoryContainer.appendChild(botMessageElement);

                // Prendiamo il <p class="message__text">
                const botTextElement = botMessageElement.querySelector(".message__text");
                // Usiamo marked per supportare il Markdown
                botTextElement.innerHTML = marked.parse(msg.text || "");

                // Evidenziamo eventuale codice
                hljs.highlightAll();

                // Aggiungiamo i bottoni "Copy" nei blocchi <pre><code>...</code></pre>
                addCopyButtonToCodeBlocks();

                // Se vuoi mostrare il pulsante di copia dell'intero messaggio, togli la classe "hide"
                const copyIcon = botMessageElement.querySelector(".message__icon");
                if (copyIcon) {
                    copyIcon.classList.remove("hide");
                }
            }
        });
    } catch (error) {
        console.error("Error loading chat history:", error);
    }

    // Nascondi l’header
    document.body.classList.add("hide-header");
};

const displayErrorMessage = (errorMessage) => {
    const errorHtml =`<div class="message__content">
        <img class="message__avatar" src="${window.STATIC_URL_LOGO}" alt="Assistant Avatar">
            <p class="message__text">${errorMessage}</p>
            </div>`;
    const errorElement = createChatMessageElement(errorHtml, "message--incoming");
    errorElement.closest(".message").classList.add("message--error");
    chatHistoryContainer.appendChild(errorElement);
};


// Call this function on page load
window.addEventListener("DOMContentLoaded", loadChatHistory);
