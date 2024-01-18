

$("select#variables-dropdown").on("change", function (e) {
    var searchText = $("select#variables-dropdown").val().toLowerCase();
    var variables = $(".variable");
    variables.each(function (index) {
        var isMatch = $(this.firstChild.firstChild).attr("title").toLowerCase() == (searchText);
        if(searchText == ""){isMatch = true};
        $(this).parent().toggle(isMatch);
    });
});


