chrome.tabs.onUpdated.addListener((tabId, tab) => {
    if (tab.url && tab.url.includes("mail.google.com/mail/u/")) {
        const queryParameters = tab.url.split("inbox/")[1];
        
        chrome.tabs.sendMessage(tabId, {
            type: "NEW",
            mailId: queryParameters,
        }).then(data=>{
            
        });
    }
});
