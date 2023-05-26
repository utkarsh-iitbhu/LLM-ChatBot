$(document).ready(function() {
    // Handle form submission
    $('#chat-form').submit(function(event) {
        event.preventDefault(); // Prevent the form from submitting

        // Get the user input from the chat input field
        var userInput = $('#chat-input').val();

        // Display the user's message in the chat container
        displayMessage('User', userInput);

        // Send the user's message to the server for processing
        $.ajax({
            url: '/process_message', // The URL to send the user's message
            type: 'POST',
            data: { message: userInput }, // The user's message data
            success: function(response) {
                // Display the chatbot's response in the chat container
                displayMessage('Chatbot', response.message);
            },
            error: function() {
                // Handle error if the request fails
                displayMessage('Error', 'An error occurred while processing the message.');
            }
        });

        // Clear the chat input field
        $('#chat-input').val('');
    });

    // Function to display a chat message in the chat container
    function displayMessage(sender, message) {
        var chatMessages = $('#chat-messages');
        var messageHTML = '<div class="message"><strong>' + sender + ':</strong> ' + message + '</div>';
        chatMessages.append(messageHTML);
        chatMessages.scrollTop(chatMessages[0].scrollHeight); // Scroll to the bottom of the chat container
    }
});