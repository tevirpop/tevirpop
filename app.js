const express = require('express');
const bodyParser = require('body-parser');
const nodemailer = require('nodemailer');

const port= 3001
const host='0.0.0.0';

app = express();

app.use(express.static(__dirname));

app.use(express.urlencoded({extended:false}));
app.use(express.json());

app.get('/', (req, res) => {
  
    res.sendFile(__dirname + '/main.html');
  })


  app.post('/submit-form', (req, res) => {
    const { name, email, comment } = req.body;
   
    const transporter = nodemailer.createTransport({
        host: 'smtp.mail.ru',
        port: '465',
        service:'mail',
        secure: true,
        auth: {
            user: 'labirynthsochi@mail.ru',
            pass: '12kL0aetgAcPxGM7kYip'
        }
      });

      const mailOptions = {
        from: 'labirynthsochi@mail.ru',
        to: email,
        subject: 'Сообщение с формы обратной связи',
       
        text: `Здравствуйте, ${name}! Спасибо за вашу заявку. Мы скоро свяжемся с вами для уточнония подробностей Данное письмо создано автоматически. Не отвечайте на него`
        
      };

      transporter.sendMail(mailOptions, (error, info) => {
        if (error) {
          console.log(error);
          res.sendFile(__dirname +'/respondbad.html');
        } else {
          
          console.log('Email sent: ' + info.response);
          res.sendFile(__dirname + '/respondgood.html');

        }
        
      });

      const mailOptions2 = {
        from: 'labirynthsochi@mail.ru',
        to: 'labirynthsochi@mail.ru',
        subject: 'Сообщение о поступлении заявки',
       
        text: `${name} оставил заявку. Комментарий: ${comment} Почта: ${email}`
        
      };
      transporter.sendMail(mailOptions2, (error, info) => {
        if (error) {
          console.log(error);
          
        } else {
          
          console.log('Email sent: ' + info.response);

        }});
    
   
  });

  /*00000000000000000000000000000000
  0000000000000000000000000000000000
  0000000000000000000000000000000000
  0000000000000000000000000000000000
  */
  app.listen(port,host, () => {
    console.log(`Example app listening on port ${host}:${port}`)
  })