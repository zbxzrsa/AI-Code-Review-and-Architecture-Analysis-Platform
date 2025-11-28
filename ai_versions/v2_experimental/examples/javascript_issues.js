// JavaScript example with some issues to detect
function calculateTotal(items) {
  var total = 0;
  for (var i = 0; i < items.length; i++) {
    total += items[i].price;
  }
  console.log('Total calculated: ' + total);
  return total;
}

function getUserData(userId) {
  // Simulating API call
  var users = {
    123: { name: 'John', email: 'john@example.com', password: 'secret123' },
    456: { name: 'Jane', email: 'jane@example.com', password: 'password456' },
  };

  return users[userId];
}

function validateEmail(email) {
  var regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return regex.test(email);
}

// Unused variable
var unusedVar = 'This is not used';

// Missing semicolons
function processData(data) {
  var result = data.map(item => {
    return item.value * 2;
  });
  return result;
}
